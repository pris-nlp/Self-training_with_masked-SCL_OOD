from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

import pandas as pd
import tensorflow as tf
from keras.backend import set_session
import os
from typing import List
import json
import re


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def load_data(root, iter):
    texts = []
    labels = []
    partition_to_n_row = {}
    root = root + '/iter_' + str(iter) + '/'
    for partition in ['train', 'valid', 'test', 'unlabeled', 'train_aug', 'unlabeled_aug', 'valid_aug']:
        with open(root + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
            if partition == 'unlabeled' and len(lines) < 1500:
                print('!' * 200)
                print('剩下的无标注数据过少，再继续标注下去或影响准确率！')
                exit()

        if 'aug' in partition:
            partition = re.findall(r"(.*)_", partition)[0]
        with open(root + partition + ".label") as fp:
                labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list=device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_score(cm):
    fs = []
    n_class = cm.shape[0]
    # calculate the F1 score and precision and recall score for each class
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)

    f = np.mean(fs).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    f_unseen = round(fs[-1], 2)
    print("Overall(macro): ", f)
    print("Seen(macro): ", f_seen)
    print("Uneen: ", f_unseen)

    return f, f_seen, f_unseen


def get_test_info(texts: pd.Series,
                  con_texts: pd.Series,
                  label: pd.Series,
                  softmax_prob: np.ndarray,
                  softmax_classes: List[str],
                  lof_result: np.ndarray = None,
                  lof_pro: np.ndarray = None,
                  gda_result: np.ndarray = None,
                  gda_classes: List[str] = None,
                  save_to_file: bool = False,
                  output_dir: str = None) -> pd.DataFrame:
    """
    Return a pd.DataFrame, including the following information for each test instances:
        - the text of the instance
        - label & masked label of the sentence
        - the softmax probability for each seen classes (sum up to 1)
        - the softmax prediction
        - the softmax confidence (i.e. the max softmax probability among all seen classes)
        - (if use lof) lof prediction result (1 for in-domain and -1 for out-of-domain)
        - (if use gda) gda mahalanobis distance for each seen classes
        - (if use gda) the gda confidence (i.e. the min mahalanobis distance among all seen classes)
    """
    df = pd.DataFrame()
    df['label'] = label
    # for idx, _class in enumerate(softmax_classes):
    #     df[f'softmax_prob_{_class}'] = softmax_prob[:, idx]
    df['softmax_prediction'] = [softmax_classes[idx] for idx in softmax_prob.argmax(axis=-1)]
    df['softmax_confidence'] = softmax_prob.max(axis=-1)
    if lof_result is not None:
        df['lof_prediction'] = lof_result
        df['lof_pro'] = lof_pro
    if gda_result is not None:
        for idx, _class in enumerate(gda_classes):
            df[f'm_dist_{_class}'] = gda_result[:, idx]
        df['gda_prediction'] = [gda_classes[idx] for idx in gda_result.argmin(axis=-1)]
        df['gda_confidence'] = gda_result.min(axis=-1)
    df['text'] = [text for text in texts]
    df['con_text'] = [con_text for con_text in con_texts]

    if save_to_file:
        df.to_csv(os.path.join(output_dir, "test_info.csv"))

    return df

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def log_pred_results(f: float,
                     f_seen: float,
                     f_unseen: float,
                     classes: List[str],
                     output_dir: str,
                     confusion_matrix: np.ndarray,
                     threshold: float = None):
    with open(os.path.join(output_dir, "results.txt"), "w") as f_out:
        f_out.write(f"Overall(macro):  {f}\nSeen(macro):  {f_seen}\n"
                    f"=====> Uneen(Experiment) <=====:  {f_unseen}\n\n"
                    f"Classes:\n{classes}\n\n"
                    f"Threshold:\n{threshold}\n\n"
                    f"Confusion matrix:\n{confusion_matrix}")
    with open(os.path.join(output_dir, "results.json"), "w") as f_out:
        json.dump({
            "f1_overall": f,
            "f1_seen": f_seen,
            "f1_unseen": f_unseen,
            "classes": classes,
            "confusion_matrix": confusion_matrix.tolist(),
            "threshold": threshold
        }, fp=f_out, ensure_ascii=False, indent=4)