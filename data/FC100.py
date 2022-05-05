import os
import random
import pandas as pd



def create_train_test(data_dir,rate=5/6):
    ls = os.listdir(data_dir)
    class_dir = [os.path.join(data_dir, path) for path in ls]
    img_dir = []
    all_num = 0
    for cla,l in zip(class_dir,ls):
        images = [l+'/'+c for c in os.listdir(cla)]
        img_dir.append(images)
        all_num += len(images)
    train_sets = []
    test_sets = []
    for i,imgds in enumerate(img_dir):
        random.shuffle(imgds)
        labels = [i] * len(imgds)
        train_num = int(rate * len(imgds))
        for i,(imgd,label) in enumerate(zip(imgds,labels)):
            if i < train_num:
                train_sets.append([imgd,label])
            else:
                test_sets.append([imgd,label])

    names = ['dir','label']
    train = pd.DataFrame(columns=names,data=train_sets)
    test = pd.DataFrame(columns=names, data=test_sets)
    train.to_csv('data/FC100/train_csv',index=None)
    test.to_csv('data/FC100/test_csv',index=None)



def main():
    data_dir = "FC100/train"  # 指向数据集的根目录
    create_train_test(data_dir)

if __name__ == '__main__':
    main()