from __future__ import print_function
  
import os
import sys
import argparse
import time
import math
import random
import copy

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from utils import TwoCropTransform, AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate
from utils import set_optimizer, save_model, log_pred_results, get_score, get_test_info
from models.Bi_LSTM import SupConResNet
from losses import SupConLoss

from utils import load_data, set_allow_growth
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--dataset', type=str, default='CLINC', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # for text ood
    parser.add_argument("--proportion", type=int, default=100,
                        help="The proportion of seen classes, range from 0 to 100.")
    parser.add_argument("--iter", type=int, default=0,
                        help="The iter epoch of self training.")
    parser.add_argument("--embedding_file", type=str,
                        default="/home/disk2/lzj2019/research/OOD_iterative/embedding/glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--seen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    parser.add_argument("--seen_classes", type=str, nargs="+", default=None,
                        help="The specific seen classes.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both", 'pseudo'], default="both",
                        help="Specify running mode: only train, only test or both.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping.")
    parser.add_argument("--ce", type=int, default=1,
                        help="Whether use the cross-entropy to train the model.")
    parser.add_argument("--ce_ind", type=int, default=0,
                        help="Use cross-entropy as the local loss of IND data and SCL for all space data.")
    parser.add_argument('--balance', type=int, default=0,
                        help='Whether to balance the pseudo data.')



    opt = parser.parse_args()


    # set the path according to the environment
    opt.model_path = './save/SupCon_ce_{}_ceind_{}_balance_{}/{}_models_iter_{}'.format(opt.ce, opt.ce_ind, opt.balance, opt.dataset, opt.iter)
    opt.tb_path = './save/SupCon_ce_{}_ceind_{}_balance_{}/{}_tensorboard_iter_{}'.format(opt.ce, opt.ce_ind, opt.balance, opt.dataset, opt.iter)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


opt = parse_option()
dataset = opt.dataset
proportion = opt.proportion
iter = opt.iter
data_root = './data/CLINC_ce_{}_ceind_{}_balance_{}'.format(opt.ce, opt.ce_ind, opt.balance)

print('*' * 100)
print('*' * 30, 'iter_{}_mode_{}_ce_{}_ce_ind_{}_balance_{}'.format(opt.iter, opt.mode, opt.ce, opt.ce_ind, opt.balance), '*' * 30)
print('*' * 100)

EMBEDDING_FILE = opt.embedding_file
MAX_SEQ_LEN = opt.max_seq_len
MAX_NUM_WORDS = opt.max_num_words
EMBEDDING_DIM = opt.embedding_dim
BATCH_SIZE = opt.batch_size


################################################################################################################
# =============================================== SET_LOADER ============================================
df, partition_to_n_row = load_data(data_root, iter)

df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
texts = df['content_words'].apply(lambda l: " ".join(l))

# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Train-valid-test split
print(partition_to_n_row)
idx_train = (None,
             partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'],
             partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'],
            partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'])
idx_unlabel = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'],
               partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'])
idx_train_aug = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'],
                 partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'] + partition_to_n_row['train_aug'])
idx_unlabel_aug = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'] + partition_to_n_row['train_aug'],
                   partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'] + partition_to_n_row['train_aug'] + partition_to_n_row['unlabeled_aug'])
idx_valid_aug = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'] + partition_to_n_row['unlabeled'] + partition_to_n_row['train_aug'] + partition_to_n_row['unlabeled_aug'],
                 None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]
X_unlabel = sequences_pad[idx_unlabel[0]:idx_unlabel[1]]
X_train_aug = sequences_pad[idx_train_aug[0]:idx_train_aug[1]]
X_unlabel_aug = sequences_pad[idx_unlabel_aug[0]:idx_unlabel_aug[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]
df_unlabel = df[idx_unlabel[0]:idx_unlabel[1]]
df_train_aug = df[idx_train_aug[0]: idx_train_aug[1]]
df_unlabel_aug = df[idx_unlabel_aug[0]: idx_unlabel_aug[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)
y_unlabel = df_unlabel.label.reset_index(drop=True)
print("train : valid : test : unlabel : train_aug : unlabel_aug = %d : %d : %d : %d : %d : %d" % (
X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_unlabel.shape[0], X_train_aug.shape[0], X_unlabel_aug.shape[0]))

n_class = y_train.unique().shape[0]

# TEST_LABEL
le_test = LabelEncoder()
le_test.fit(y_test)

# TRAIN_LABEL
le = LabelEncoder()
le.fit(y_train)
y_train_idx = le.transform(y_train)
y_valid_idx = le.transform(y_valid)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(X_train_aug), torch.tensor(y_train_idx))
valid_dataset = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid_idx))
#创建DataLoader迭代器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)



################################################################################################################
# =============================================== LOAD_EMBEDDING ============================================
print("Load pre-trained GloVe embedding...")
MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector




################################################################################################################
# =============================================== SET_MODEL ====================================================
model = SupConResNet(embedding_matrix=embedding_matrix, feat_dim=len(list(le.classes_)))
print(model)
ce_criterion = torch.nn.CrossEntropyLoss()
criterion = SupConLoss(temperature=opt.temp)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    ce_criterion = ce_criterion.cuda()
    cudnn.benchmark = True


#############################################################################################################
# =========================================== TRAIN_EACH_EPOCH  ============================================
def convert_tensor_to_numpy(tensor_list):
    return [tensor.cpu().detach().numpy() for tensor in tensor_list]

def train(train_loader, model, criterion, ce_criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (text, con_text, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        texts = torch.cat([text, con_text], dim=0).type(torch.LongTensor)
        ce_texts = text.type(torch.LongTensor)

        if torch.cuda.is_available():
            texts = texts.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ce_texts = ce_texts.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, logits, features_base, _ = model(texts)
        _, logits_ce, _, _ = model(ce_texts)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)
        if opt.ce == 1:
            loss = ce_criterion(logits_ce, labels)
        # delete oos relative rows
        if opt.ce_ind == 1:
            p = (labels == le_test.transform(np.array(['oos']))[0]).nonzero()
            if opt.iter != 0 and p.shape[0] > 0:
                for row in p:
                    logits_ce = logits_ce[torch.arange(logits_ce.size(0)) != row]
                    labels = labels[torch.arange(labels.size(0)) != row]
            loss = ce_criterion(logits_ce, labels) + loss

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)
        # exit()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses))
        #     sys.stdout.flush()


    return losses.avg


################################################################################################################
#=========================================== TRAIN_AND_EVAL_TOTAL_EPOCH  ============================================
# add eval
if opt.mode == 'train':
    optimizer = set_optimizer(opt, model)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)


    valid_losses = AverageMeter()
    valid_criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    min_loss = 1000
    best_model = None
    best_flag = 0
    last_loss = 1000
    patience = opt.patience

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, criterion, ce_criterion, optimizer, epoch, opt)

        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # if epoch % 3 == 0:
        #     best_flag = 1
        #     model.eval()
        #     valid_texts = torch.tensor(X_valid_seen).type(torch.LongTensor)
        #
        #     if torch.cuda.is_available():
        #         valid_texts = valid_texts.cuda(non_blocking=True)
        #
        #     valid_features, valid_probs, valid_features_lof = model(valid_texts)
        #     [valid_probs, valid_features_lof] = convert_tensor_to_numpy([valid_probs, valid_features_lof])
        #     result = pd.DataFrame(valid_probs).idxmax(axis=1)
        #
        #     f1 = metrics.f1_score(y_valid_idx, result, average='macro')
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         best_model = copy.deepcopy(model)
        #     print('current_f1:{f1:.4f}'.format(f1=f1))


        model.eval()

        for texts, labels in valid_loader:
            texts = texts.type(torch.LongTensor)
            if torch.cuda.is_available():
                texts = texts.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            _, logits, _, _ = model(texts)
            vloss = valid_criterion(logits, labels)
            valid_losses.update(vloss.item(), labels.shape[0])

        avg_valid_loss = valid_losses.avg
        valid_losses.reset()

        print('epoch {}, train_loss {:.4f}, valid_loss {:.4f}, patience {}, learning rate {:.4f}'.format(
            epoch, loss, avg_valid_loss, patience, optimizer.param_groups[0]['lr']))

        if avg_valid_loss <= min_loss:
            best_flag = 1
            min_loss = avg_valid_loss
            best_model = copy.deepcopy(model)

        patience = opt.patience if avg_valid_loss < last_loss else patience - 1
        last_loss = avg_valid_loss
        if patience == 0:
            print('early stopping......: last loss is:', last_loss, 'min loss is:', min_loss)
            break


    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    if best_flag:
        model = best_model
    save_model(model, optimizer, opt, opt.epochs, save_file)



################################################################################################################
#=========================================== TEST_AND_PSEUDO  ============================================
def test(train_data, test_data, test_labels, model, opt):
    model.eval()
    classes = list(le_test.classes_)
    print('测试集标签一共 %d 个' % len(classes))

    train_texts = torch.tensor(train_data).type(torch.LongTensor)
    test_texts = torch.tensor(test_data).type(torch.LongTensor)

    if torch.cuda.is_available():
        train_texts = train_texts.cuda(non_blocking=True)
        test_texts = test_texts.cuda(non_blocking=True)

    train_features, _, train_features_lof, _ = model(train_texts)
    test_features, _, test_features_lof, test_probs = model(test_texts)

    train_features, train_features_lof, test_features, test_probs, test_features_lof = convert_tensor_to_numpy([train_features, train_features_lof, test_features, test_probs, test_features_lof])

    pred_dir = os.path.join(opt.model_path, 'pred')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
    lof.fit(train_features_lof)
    y_pred_lof = pd.Series(lof.predict(test_features_lof))
    y_pred_pro = pd.Series(lof.score_samples(test_features_lof))

    if iter == 0:
        df_seen = pd.DataFrame(test_probs, columns=le.classes_)
    else:
        df_seen = pd.DataFrame(test_probs, columns=classes)
    # init prediction
    df_seen['oos'] = 0

    y_pred = df_seen.idxmax(axis=1)
    y_pred[y_pred_lof[y_pred_lof == -1].index] = 'oos'

    if opt.mode == 'test':
        cm = confusion_matrix(test_labels, y_pred)
        f, f_seen, f_unseen = get_score(cm)
        print(cm)
        log_pred_results(f, f_seen, f_unseen, classes, pred_dir, cm)

    if opt.mode == 'pseudo':
        print('creating pseudo labels and store in file')
        classes_ = list(le.classes_) if iter == 0 else classes
        test_info = get_test_info(texts=texts[idx_unlabel[0]:idx_unlabel[1]],
                                  con_texts=texts[idx_unlabel_aug[0]:idx_unlabel_aug[1]],
                                  label=y_unlabel,
                                  softmax_prob=test_probs,
                                  softmax_classes=classes_,
                                  lof_result=y_pred_lof,
                                  lof_pro=y_pred_pro,
                                  save_to_file=True,
                                  output_dir=pred_dir)


if opt.mode in ['test', 'pseudo']:
    model.load_state_dict(torch.load(os.path.join(opt.save_folder, 'last.pth'))['model'])
    (x, y) = (X_test, y_test) if opt.mode == 'test' else (X_unlabel, y_unlabel)
    test(X_train, x, y, model, opt)


