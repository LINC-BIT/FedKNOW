import functools
import os
import random

import pandas as pd

def create_train(data_dir,rate=5/6):
    ls = os.listdir(data_dir)
    label_dict = {}
    for i,label in enumerate(ls):
        label_dict[label] = i
    class_dir = [os.path.join(data_dir, path) for path in ls]
    img_dir = []
    all_num = 0
    for cla,l in zip(class_dir,ls):
        images = [l+'/images/'+c for c in os.listdir(cla+'/images')]
        img_dir.append(images)
        all_num += len(images)
    train_sets = []

    for i,imgds in enumerate(img_dir):
        random.shuffle(imgds)
        labels = [i] * len(imgds)
        for i,(imgd,label) in enumerate(zip(imgds,labels)):
            train_sets.append([imgd,label])

    names = ['dir','label']
    train = pd.DataFrame(columns=names,data=train_sets)
    train.to_csv('tiny-imagenet-200/train.csv',index=None)
    return label_dict
def cmp_ignore_case(s1, s2):
    t1=s1[1]
    t2=s2[1]
    if(t1>t2):
        return 1
    if(t1==t2):
        return 0
    return -1
def create_test(data_dir,label_dict):
    test_sets = []
    names = ['dir', 'label']
    with open(os.path.join(data_dir, 'val_annotations.txt'), 'r') as fo:
        entry = fo.readlines()
        for data in entry:
            words = data.split("\t")
            test_sets.append(['images/'+words[0],label_dict[words[1]]])
    test_sets.sort(key= functools.cmp_to_key(cmp_ignore_case))
    test = pd.DataFrame(columns=names, data=test_sets)
    test.to_csv('tiny-imagenet-200/test.csv', index=None)


def main():
    data_dir = "tiny-imagenet-200/train"  # 指向数据集的根目录
    test_dir = "tiny-imagenet-200/val"
    label_dict = create_train(data_dir)
    create_test(test_dir,label_dict)


if __name__ == '__main__':
    main()