import tqdm
import json
import argparse

from back_translate import back_translate

def get_translate_dict(input_path):
    schema_path = './input/schemas.json'
    schemas = json.load(open(schema_path, "r"))
    with open(args.input_file, "r") as f_in:
        lines = [line.strip() for line in f_in.readlines() if line.strip()]

    for line in tqdm.tqdm(lines):
        ques, ans = line.split("\t")
        result_dict = back_translate(ques, schemas, None)
    return result_dict


def main(args):
    schemas = json.load(open(args.schema_file, "r"))

    fail_aug = []
    full_data_dict = {}
    cnt = 0
    with open('../data/CLINC/data_full.json') as f,\
        open(args.output_file, "w") as f_out:
        data_dic = json.load(f)
        for key in data_dic.keys():
            for example in data_dic[key]:
                result_dict = back_translate(example[0], schemas, None)
                string = []
                for key in result_dict.keys():
                    string.append(result_dict[key])

                if len(string) != 3:
                    fail_aug.append(example)
                    continue
                string.append(example[1])
                full_data_dict[cnt] = string
                cnt += 1
                if cnt % 100 == 0:
                    print('total 23700, now is:', cnt)
        json.dump(full_data_dict, f_out, indent=4)

    # {
    #     "0": [
    #         "set a warning for when my bank account starts running low",
    #         "My bank account is low",
    #         "Warn me if my bank account runs out.",
    #         "oos"
    #     ],
    #     "1": [
    #         "a show on broadway",
    #         "Broadway show",
    #         "a show on Broadway",
    #         "oos"
    #     ]
    # }

    if fail_aug:
        with open('output/fail_aug.txt', "w") as f_fail:
            for example in fail_aug:
                f_fail.write(example[0] + ',' + example[1] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=False, default='output/full_result.json')
    parser.add_argument("--schema_file", type=str, required=False, default='/home/disk2/lzj2019/research/OOD_iterative/scl_text_classification/back_translate/input/schemas.json')
    parser.add_argument("--keywords_file", type=str, help="Required if keyword mask is applied")
    args = parser.parse_args()

    main(args)
