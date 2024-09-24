import torch
import os

def split_data(num_client,train_file,test_file):
    client_dict = './data/amazon_book/data_' + str(num_client) + 'clients'
    num_train = 1086120 // num_client
    num_test = 121216 // num_client

    if not os.path.exists(client_dict):
        os.mkdir(client_dict)

    f_train = open(train_file, 'r')
    f_test = open(test_file, 'r')

    for i in range(num_client):
        file = client_dict + '/train_data_' + str(i)
        with open(file, 'a') as f:
            for j in range(num_train):
                a = f_train.readline()
                f.write(a)

    for i in range(num_client):
        file = client_dict + '/test_data_' + str(i)
        with open(file, 'a') as f:
            for j in range(num_test):
                a = f_test.readline()
                f.write(a)


class DINDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, user_map, material_map, category_map, max_length):

        user = [];
        material = [];
        category = []
        material_historical = [];
        category_historical = []
        material_historical_neg = [];
        category_historical_nge = []
        mask = [];
        sequential_length = []
        target = []

        with open(data_path, 'r') as fin:

            for line in fin:
                item = line.strip('\n').split('\t')
                if not item: continue

                user.append(user_map.get(item[1], 0))
                material.append(material_map.get(item[2], 0))
                category.append(category_map.get(item[3], 0))

                material_historical_item = [0] * max_length
                temp = item[4].split("")
                if (len(temp) >= max_length): temp = temp[-max_length:]
                for i, m in enumerate(temp):
                    material_historical_item[i] = material_map.get(m, 0)
                material_historical.append(material_historical_item)

                category_historical_item = [0] * max_length
                temp = item[5].split("")
                if (len(temp) >= max_length): temp = temp[-max_length:]
                for i, c in enumerate(temp):
                    category_historical_item[i] = category_map.get(c, 0)
                category_historical.append(category_historical_item)

                temp = min(len(temp), max_length)
                mask_item = [1] * temp + [0] * (max_length - temp)

                mask.append(mask_item)
                sequential_length.append(temp)

                target.append(int(item[0]))

        self.user = torch.tensor(user)

        self.material = torch.tensor(material)
        self.catetory = torch.tensor(category)

        self.material_historical = torch.tensor(material_historical)
        self.catetory_historical = torch.tensor(category_historical)

        self.mask = torch.tensor(mask).type(torch.FloatTensor)
        self.sequential_length = torch.tensor(sequential_length)

        self.target = torch.tensor(target)

    def __len__(self):
        return len(self.user)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        user = self.user[index]

        material_historical = self.material_historical[index, :]
        category_historical = self.catetory_historical[index, :]
        mask = self.mask[index, :]
        sequential_length = self.sequential_length[index]

        material = self.material[index]
        category = self.catetory[index]

        target = self.target[index]

        return user, material_historical, category_historical, mask, sequential_length, \
               material, category, 0, 0, target