import os
import json
import random

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def create_train_test(data_dir,rate=0.8):
    ls = os.listdir(data_dir+'/core50_128x128')
    csv_dir = data_dir+'/'+'task_label/'
    for i in range(len(ls)):
        if not os.path.exists(csv_dir+str(i)):
            os.mkdir(csv_dir+str(i))
    tasks_dir = [os.path.join(data_dir+'/core50_128x128', path) for path in ls]
    all_dir = []
    for task in tasks_dir:
        temp = [os.path.join(task,dir) for dir in os.listdir(task)]
        all_dir.append(temp)

    for t,class_dir in enumerate(all_dir):
        img_dir = []
        all_num = 0
        for cla in class_dir:
            images = [os.path.join(cla,c) for c in os.listdir(cla)]
            img_dir.append(images)
            if len(images)<300:
                print(images[0])
            all_num += len(images)
        train_sets = []
        test_sets = []
        for i,imgds in enumerate(img_dir):
            random.shuffle(imgds)
            labels = [50*t+i] * len(imgds)
            train_num = int(rate * len(imgds))
            for j,(imgd,label) in enumerate(zip(imgds,labels)):
                if j < train_num:
                    train_sets.append([imgd,label])
                else:
                    test_sets.append([imgd,label])

        names = ['dir','label']
        train = pd.DataFrame(columns=names,data=train_sets)
        test = pd.DataFrame(columns=names, data=test_sets)
        train.to_csv(os.path.join(csv_dir+str(t),'train.csv'),index=None)
        test.to_csv(os.path.join(csv_dir+str(t),'test.csv'),index=None)



def main():
    data_dir = "core50"  # 指向数据集的根目录
    create_train_test(data_dir)


if __name__ == '__main__':
    main()