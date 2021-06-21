import csv
import os
import random
import argparse
import json
 
def parse_args():
    parser = argparse.ArgumentParser()
    # arguments need to specify
    parser.add_argument("--file_path", type=str, required=False,
                        help="The path of 'test_info.csv' file.")
    parser.add_argument("--iter", type=int, required=True,
                        help="The iter epoch of self_training.")
    parser.add_argument("--balance", type=int, default=0,
                        help="Whether to balance the num of each class.")
    parser.add_argument("--ce", type=int, default=1,
                        help="Whether use the cross-entropy to train the model.")
    parser.add_argument("--ce_ind", type=int, default=0,
                        help="Use cross-entropy as the local loss of IND data and SCL for all space data.")
    parser.add_argument('--dataset', type=str, default='CLINC', help='dataset')

    args = parser.parse_args()
    return args

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_dic_from_json():
    cate_dic = {}
    cnt = 0
    with open('back_translate/output/full_result.json') as f:
        data_dic = json.load(f)
        for index in data_dic.keys():
            example = data_dic[index]
            if example[3] not in cate_dic:
                cate_dic[example[3]] = []
            cate_dic[example[3]].append(example[0: 3])
            cnt += 1
    return cate_dic, cnt

def split(root):
    seed = SEED
    file_path = root + '/iter_0'
    create_dir(file_path)
    with open(file_path + '/train.seq.in', 'w', encoding='utf-8') as labeled, \
        open(file_path + '/train.label', 'w', encoding='utf-8') as labeled_label, \
        open(file_path + '/test.seq.in', 'w', encoding='utf-8') as test_f, \
        open(file_path + '/test.label', 'w', encoding='utf-8') as test_f_label, \
        open(file_path + '/unlabeled.seq.in', 'w', encoding='utf-8') as unlabeled, \
        open(file_path + '/unlabeled.label', 'w', encoding='utf-8') as unlabeled_label, \
        open(file_path + '/valid.seq.in', 'w', encoding='utf-8') as valid, \
        open(file_path + '/valid.label', 'w', encoding='utf-8') as valid_label, \
        open(file_path + '/train_aug.seq.in', 'w', encoding='utf-8') as train_aug,\
        open(file_path + '/unlabeled_aug.seq.in', 'w', encoding='utf-8') as unlabel_aug,\
        open(file_path + '/valid_aug.seq.in', 'w', encoding='utf-8') as valid_aug:

        data_dic, cnt = get_dic_from_json()
        print('total number is ', cnt)

        ind_total_each_class = 150
        ood_total_one_class = 1200

        random.seed(seed)  # set random seed for consistent result
        ind_index = list(range(ind_total_each_class))
        random.shuffle(ind_index)

        random.seed(seed)
        ood_index = list(range(ood_total_one_class))
        random.shuffle(ood_index)

        ood_unlabel = 700
        ind_train = 24
        ind_valid = 6
        ind_unlabel = 70
        # ind_test = 150 - (ind_valid + ind_train + ind_unlabel)

        for key in data_dic.keys():
            # if example is OOD: 700 for unlabel expansion; 500 for test
            if 'oos' in key:
                for i in range(ood_unlabel):
                    unlabeled.write(data_dic[key][ood_index[i]][0] + '\n')
                    unlabel_aug.write(data_dic[key][ood_index[i]][AUG_select] + '\n')
                    unlabeled_label.write(key + '\n')
                for i in range(ood_unlabel, len(data_dic[key])):
                    test_f.write(data_dic[key][ood_index[i]][0] + '\n')
                    test_f_label.write(key + '\n')
            # if example is IND: 24 for train and 6 valid; 70 for unlabel expansion; 50 for test
            else:
                for i in range(ind_train):
                    labeled.write(data_dic[key][ind_index[i]][0] + '\n')
                    train_aug.write(data_dic[key][ind_index[i]][AUG_select] + '\n')
                    labeled_label.write(key + '\n')
                for i in range(ind_train, ind_train + ind_valid):
                    valid.write(data_dic[key][ind_index[i]][0] + '\n')
                    valid_aug.write(data_dic[key][ind_index[i]][AUG_select] + '\n')
                    valid_label.write(key + '\n')
                for i in range(ind_train + ind_valid, ind_valid + ind_train + ind_unlabel):
                    unlabeled.write(data_dic[key][ind_index[i]][0] + '\n')
                    unlabel_aug.write(data_dic[key][ind_index[i]][AUG_select] + '\n')
                    unlabeled_label.write(key + '\n')
                for i in range(ind_valid + ind_train + ind_unlabel, len(data_dic[key])):
                    test_f.write(data_dic[key][ind_index[i]][0] + '\n')
                    test_f_label.write(key + '\n')


def read_file_to_list(file_path):
    re = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            re.append(line.strip())
    return re

def read_train_and_valid_to_dic(train_seq, train_label, train_aug_seq, val_seq, val_label, val_aug_seq):
    '''
    得到上一轮迭代使用的所有训练+验证数据，以类别为key加入字典中
    Args:
        train_seq:
        train_label:
        val_seq:
        val_label:

    Returns:

    '''
    dic = {}
    count = 0
    with open(train_seq, 'r', encoding='utf-8') as ts,\
        open(train_label, 'r', encoding='utf-8') as tl,\
        open(train_aug_seq, 'r', encoding='utf-8') as tas,\
        open(val_seq, 'r', encoding='utf-8') as vs,\
        open(val_label, 'r', encoding='utf-8') as vl,\
        open(val_aug_seq, 'r', encoding='utf-8') as vas:
        ts_lines = ts.read().splitlines()
        tl_lines = tl.read().splitlines()
        tas_lines = tas.read().splitlines()
        vs_lines = vs.read().splitlines()
        vl_lines = vl.read().splitlines()
        vas_lines = vas.read().splitlines()
        if len(ts_lines) != len(tl_lines) or len(vs_lines) != len(vl_lines) or len(ts_lines) != len(tas_lines) or len(vs_lines) != len(vas_lines):
            print('上一轮数据出现了错误，请检查样本和标签是否对齐！')

        for i in range(len(ts_lines)):
            if tl_lines[i] not in dic:
                dic[tl_lines[i]] = []
            count += 1
            dic[tl_lines[i]].append((ts_lines[i], tas_lines[i]))

        for j in range(len(vs_lines)):
            count += 1
            dic[vl_lines[j]].append((vs_lines[j], vas_lines[j]))
    print('The number of total data is ', count)
    return dic


def change_list_to_file(data_list, new_path):
    with open(new_path, 'w', encoding='utf-8') as f:
        for example in data_list:
            f.write(example + '\n')

def copy_file(old_path, new_path):
    with open(old_path, 'r', encoding='utf-8') as of,\
        open(new_path, 'w', encoding='utf-8') as nf:
        lines = of.readlines()
        for line in lines:
            nf.write(line)

def add_info(iter, new_train, new_val, unlabel_leave, add_oos, file_path):
    dic = {
        'iter': iter,
        'new_train': new_train,
        'new_val': new_val,
        'unlabel_leave': unlabel_leave,
        'add_oos': add_oos
    }
    with open(file_path, 'w') as f:
        json.dump(dic, f)

def add_pseudo(root, file_path, iter):
    '''

    Args:
        file_path: /home/disk2/lzj2019/research/OOD_iterative/OOD-master/outputs/models/CLINC-100-1607515498.7126565/lof/test_info.csv
        iter: 当前迭代轮次，用于创建新的训练文件

    Returns:

    '''
    unseen_label = 'oos'
    oos_gate = float(-1)
    train_valid_split = 0.2 # valid count 20% in total examples, and train for 80%


    root = root + '/'
    if iter == 0:
        return 'init teacher model no need to merge dataset!'
    else:
        print('now is the %d th iteration' %iter)

    last_iter_file = root + 'iter_' + str(iter - 1)
    new_iter_file = root + 'iter_' + str(iter)
    create_dir(new_iter_file)
    if not os.path.exists(last_iter_file):
        print('last iter dataset missed ~~~')

    old_train_seq = last_iter_file + '/train.seq.in'
    old_train_label = last_iter_file + '/train.label'
    old_train_aug_seq = last_iter_file + '/train_aug.seq.in'
    old_valid_seq = last_iter_file + '/valid.seq.in'
    old_valid_label = last_iter_file + '/valid.label'
    old_valid_aug_seq = last_iter_file + '/valid_aug.seq.in'
    old_test_seq = last_iter_file + '/test.seq.in'
    old_test_label = last_iter_file + '/test.label'

    new_train_seq = new_iter_file + '/train.seq.in'
    new_train_label = new_iter_file + '/train.label'
    new_train_aug_seq = new_iter_file + '/train_aug.seq.in'
    new_valid_seq = new_iter_file + '/valid.seq.in'
    new_valid_label = new_iter_file + '/valid.label'
    new_valid_aug_seq = new_iter_file + '/valid_aug.seq.in'
    new_unlabel_seq = new_iter_file + '/unlabeled.seq.in'
    new_unlabel_label = new_iter_file + '/unlabeled.label'
    new_unlabel_aug_seq = new_iter_file + '/unlabeled_aug.seq.in'
    new_test_seq = new_iter_file + '/test.seq.in'
    new_test_label = new_iter_file + '/test.label'

    new_data_info = new_iter_file + '/info.json'

    copy_file(old_test_seq, new_test_seq)
    copy_file(old_test_label, new_test_label)

    total_data = read_train_and_valid_to_dic(old_train_seq, old_train_label, old_train_aug_seq, old_valid_seq, old_valid_label, old_valid_aug_seq)
    # print('total data', total_data.keys())
    if unseen_label not in total_data.keys():
        print('上一轮的训练数据中没有标签 oos，这一轮添加进去')
        total_data[unseen_label] = []

    new_data = {
           'train.seq': [],
           'train.label': [],
           'train_aug.seq': [],
           'valid.seq': [],
           'valid.label': [],
           'valid_aug.seq': [],
           'unlabel.seq':[],
           'unlabel.label':[],
           'unlabel_aug.seq': []
           }

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # obtain the gate score for ind data
        ind_score = []
        for ele in reader:
            ind_score.append(float(ele['softmax_confidence']))
        # ind 阈值之后可以设置和迭代轮次相关
        ind_gate = float(min(ind_score) + (4 / 7) * (max(ind_score) - min(ind_score)))
        print('ind field is:', ind_gate)

        # 这一版是直接添加，之后可以先排序再截取，对数据做 balance(各类之间保持平衡) ，且每轮的截取数目都和轮次有关
    with open(file_path, 'r', encoding='utf-8') as new_f:
        reader = csv.DictReader(new_f)
        for example in reader:
            if int(example['lof_prediction']) == -1 and float(example['lof_pro']) < oos_gate:
                total_data[unseen_label].append((example['text'], example['con_text']))
            elif int(example['lof_prediction']) == 1 and float(example['softmax_confidence']) > ind_gate:
                total_data[example['softmax_prediction']].append((example['text'], example['con_text']))
            else:
                new_data['unlabel.seq'].append(example['text'])
                new_data['unlabel.label'].append(example['label'])
                new_data['unlabel_aug.seq'].append(example['con_text'])

        print('oos number is: ', len(total_data[unseen_label]))
        count_oos = 0
        for key in total_data.keys():
            total_train_num = len(total_data[key])

            random.seed(SEED)
            index = list(range(total_train_num))
            random.shuffle(index)

            for i in range(int(total_train_num * train_valid_split)):
                if key == unseen_label:
                    count_oos += 1
                new_data['valid.seq'].append(total_data[key][index[i]][0])
                new_data['valid.label'].append(key)
                new_data['valid_aug.seq'].append(total_data[key][index[i]][1])

            for i in range(int(total_train_num * train_valid_split), total_train_num):
                if key == unseen_label:
                    count_oos += 1
                new_data['train.seq'].append(total_data[key][index[i]][0])
                new_data['train.label'].append(key)
                new_data['train_aug.seq'].append(total_data[key][index[i]][1])


        print('新的训练数据一共:', len(new_data['train.seq']), '新的验证数据一共:', len(new_data['valid.seq']),
              '剩下的无标注数据一共:', len(new_data['unlabel.seq']), '新加入的 oos 数据一共：', count_oos)

        add_info(iter, len(new_data['train.seq']), len(new_data['valid.seq']), len(new_data['unlabel.seq']), count_oos, new_data_info)

        change_list_to_file(new_data['train.seq'], new_train_seq)
        change_list_to_file(new_data['train.label'], new_train_label)
        change_list_to_file(new_data['train_aug.seq'], new_train_aug_seq)

        change_list_to_file(new_data['valid.seq'], new_valid_seq)
        change_list_to_file(new_data['valid.label'], new_valid_label)
        change_list_to_file(new_data['valid_aug.seq'], new_valid_aug_seq)

        change_list_to_file(new_data['unlabel.seq'], new_unlabel_seq)
        change_list_to_file(new_data['unlabel.label'], new_unlabel_label)
        change_list_to_file(new_data['unlabel_aug.seq'], new_unlabel_aug_seq)


def add_pseudo_with_balance(root, file_path, iter):
    '''
    以得到pseudo 标注最少数量样本为基础，新加入的样本保证和原来样本一起算下来每一类的数目一样。具体操作是：先得到新标注的IND数据，
    找到最小数目min-num，先对IND数据根据min-num做balance，之后对所有新标注的oos数据排序，取前（min-num+original-ind - original—oos）
    个加入训练集。最后每一类包括oos数据都是 min-num
    Args:
        file_path: /home/disk2/lzj2019/research/OOD_iterative/OOD-master/outputs/models/CLINC-100-1607515498.7126565/lof/test_info.csv
        iter: 当前迭代轮次，用于创建新的训练文件

    Returns:

    '''
    unseen_label = 'oos'
    train_valid_split = 0.2 # valid count 20% in total examples, and train for 80%


    root = root + '/'
    if iter == 0:
        return 'init teacher model no need to merge dataset!'
    else:
        print('now is the %d th iteration' %iter)

    last_iter_file = root + 'iter_' + str(iter - 1)
    new_iter_file = root + 'iter_' + str(iter)
    create_dir(new_iter_file)
    if not os.path.exists(last_iter_file):
        print('last iter dataset missed ~~~')

    old_train_seq = last_iter_file + '/train.seq.in'
    old_train_label = last_iter_file + '/train.label'
    old_train_aug_seq = last_iter_file + '/train_aug.seq.in'
    old_valid_seq = last_iter_file + '/valid.seq.in'
    old_valid_label = last_iter_file + '/valid.label'
    old_valid_aug_seq = last_iter_file + '/valid_aug.seq.in'
    old_test_seq = last_iter_file + '/test.seq.in'
    old_test_label = last_iter_file + '/test.label'

    new_train_seq = new_iter_file + '/train.seq.in'
    new_train_label = new_iter_file + '/train.label'
    new_train_aug_seq = new_iter_file + '/train_aug.seq.in'
    new_valid_seq = new_iter_file + '/valid.seq.in'
    new_valid_label = new_iter_file + '/valid.label'
    new_valid_aug_seq = new_iter_file + '/valid_aug.seq.in'
    new_unlabel_seq = new_iter_file + '/unlabeled.seq.in'
    new_unlabel_label = new_iter_file + '/unlabeled.label'
    new_unlabel_aug_seq = new_iter_file + '/unlabeled_aug.seq.in'
    new_test_seq = new_iter_file + '/test.seq.in'
    new_test_label = new_iter_file + '/test.label'

    new_data_info = new_iter_file + '/info.json'

    copy_file(old_test_seq, new_test_seq)
    copy_file(old_test_label, new_test_label)

    total_data = read_train_and_valid_to_dic(old_train_seq, old_train_label, old_train_aug_seq, old_valid_seq,
                                             old_valid_label, old_valid_aug_seq)

    total_classes = []
    for key in total_data.keys():
        total_classes.append(key)

    ORIGINAL_IND = len(total_data[total_classes[0]])
    ORIGINAL_OOS = len(total_data[unseen_label]) if unseen_label in total_classes else 0

    # print('total data', total_data.keys())
    if unseen_label not in total_classes:
        print('上一轮的训练数据中没有标签 oos，这一轮添加进去')
        total_data[unseen_label] = []
        total_classes.append(unseen_label)

    new_data = {
        'train.seq': [],
        'train.label': [],
        'train_aug.seq': [],
        'valid.seq': [],
        'valid.label': [],
        'valid_aug.seq': [],
        'unlabel.seq': [],
        'unlabel.label': [],
        'unlabel_aug.seq': []
    }

    pseudo_data = {}
    for key in total_classes:
        pseudo_data[key] = []

    with open(file_path, 'r', encoding='utf-8') as new_f:
        reader = csv.DictReader(new_f)
        for example in reader:
            if int(example['lof_prediction']) == -1:
                pseudo_data[unseen_label].append(((example['text'], example['con_text']), example['lof_pro']))
            elif int(example['lof_prediction']) == 1:
                pseudo_data[example['softmax_prediction']].append(((example['text'], example['con_text']), example['softmax_confidence']))
            else:
                assert int(example['lof_prediction']) not in [-1, 1], 'The LOF score of example occur fault!'

    min_num = 0
    len_list = []
    for cate in total_classes:
        if cate != unseen_label:
            # min_num = min(len(pseudo_data[cate]), min_num)
            len_list.append(len(pseudo_data[cate]))
    # min_num = min(min_num, 30)
    len_list = sorted(len_list)
    min_num = min(len_list[int(len(len_list) / 2)], ADD_ITER)

    num_oos = min_num + ORIGINAL_IND - ORIGINAL_OOS
    print('IND 每类添加 %d 个样本，OOD 添加 %d 个样本' % (min_num, num_oos))


    for cate in total_classes:
        num_cur = len(pseudo_data[cate])
        if cate != unseen_label:
            tmp = sorted(pseudo_data[cate], key=lambda x: x[1], reverse=True)

            if num_cur < min_num:
                bias = min_num - num_cur
                random.seed(SEED)
                index = list(range(len(total_data[cate])))
                random.shuffle(index)
                choice = index[:bias]
                tmp.append(total_data[cate][index] for index in choice)
            # print(type([x[0] for x in tmp]))
            total_data[cate].extend([x[0] for x in tmp][: min_num])
            new_data['unlabel.seq'].extend([x[0][0] for x in tmp][min_num: ])
            new_data['unlabel_aug.seq'].extend([x[0][1] for x in tmp][min_num:])
            new_data['unlabel.label'].extend([cate]* (len(pseudo_data[cate]) - min_num))
        else:
            # oos 的分数越小置信度越高
            tmp = sorted(pseudo_data[cate], key=lambda x: x[1])
            if num_cur < num_oos:
                bias = num_oos - num_cur
                random.seed(SEED)
                index = list(range(len(total_data[cate])))
                random.shuffle(index)
                choice = index[:bias]
                tmp.append(total_data[cate][index] for index in choice)

            total_data[cate].extend([x[0] for x in tmp][: num_oos])
            new_data['unlabel.seq'].extend([x[0][0] for x in tmp][num_oos:])
            new_data['unlabel_aug.seq'].extend([x[0][1] for x in tmp][num_oos:])
            new_data['unlabel.label'].extend([cate] * (len(pseudo_data[cate]) - num_oos))

    count_oos = 0
    for key in total_data.keys():
        total_train_num = len(total_data[key])

        random.seed(SEED)
        index = list(range(total_train_num))
        random.shuffle(index)

        for i in range(int(total_train_num * train_valid_split)):
            if key == unseen_label:
                count_oos += 1
            new_data['valid.seq'].append(total_data[key][index[i]][0])
            new_data['valid.label'].append(key)
            new_data['valid_aug.seq'].append(total_data[key][index[i]][1])

        for i in range(int(total_train_num * train_valid_split), total_train_num):
            if key == unseen_label:
                count_oos += 1
            new_data['train.seq'].append(total_data[key][index[i]][0])
            new_data['train.label'].append(key)
            new_data['train_aug.seq'].append(total_data[key][index[i]][1])

    print('新的训练数据一共:', len(new_data['train.seq']), '新的验证数据一共:', len(new_data['valid.seq']),
          '剩下的无标注数据一共:', len(new_data['unlabel.seq']), '新加入的 oos 数据一共：', count_oos)

    add_info(iter, len(new_data['train.seq']), len(new_data['valid.seq']), len(new_data['unlabel.seq']), count_oos,
             new_data_info)

    change_list_to_file(new_data['train.seq'], new_train_seq)
    change_list_to_file(new_data['train.label'], new_train_label)
    change_list_to_file(new_data['train_aug.seq'], new_train_aug_seq)

    change_list_to_file(new_data['valid.seq'], new_valid_seq)
    change_list_to_file(new_data['valid.label'], new_valid_label)
    change_list_to_file(new_data['valid_aug.seq'], new_valid_aug_seq)

    change_list_to_file(new_data['unlabel.seq'], new_unlabel_seq)
    change_list_to_file(new_data['unlabel.label'], new_unlabel_label)
    change_list_to_file(new_data['unlabel_aug.seq'], new_unlabel_aug_seq)

def generate_fake_file(original_file, target_path):
    copy_file(original_file, target_path)

if __name__ == '__main__':
    args = parse_args()

    SEED = 2
    AUG_select = 2  # 1 or 2
    ROOT = './data/CLINC_ce_{}_ceind_{}_balance_{}'.format(args.ce, args.ce_ind, args.balance)
    ADD_ITER = 5
    model_path = './save/SupCon_ce_{}_ceind_{}_balance_{}/{}_models_iter_{}'.format(args.ce, args.ce_ind, args.balance,
                                                                                       args.dataset, args.iter-1)
    if args.iter == 0:
        split(ROOT)
    else:
        file_path = model_path + '/pred/test_info.csv'
        if args.balance == 1:
            add_pseudo_with_balance(ROOT, file_path, args.iter)
        else:
            add_pseudo(ROOT, file_path, args.iter)

'''
第一次迭代运行 split 生成 
train，train—label，
test，test—label，
valid，valid-label，
unlabel，unlabel-label，
unlabel-aug
valid_aug
train-aug
11个文件


'''
