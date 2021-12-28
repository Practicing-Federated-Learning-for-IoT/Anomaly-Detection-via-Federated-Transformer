import numpy as np
import csv
import random
import torch
from torch.utils.data import DataLoader,TensorDataset

class Loader():
    def __init__(self,arg):
        self.arg = arg
        self.client_dataset = []
        self.test = []
        self.test_label = np.array([])

    def dataLoading(self):
        path = self.arg.path + self.arg.dataset
        # loading data
        x = []
        labels = []
        with (open(path, 'r')) as data_from:
            csv_reader = csv.reader(data_from)
            for i in csv_reader:
                x.append(i[0:self.arg.d_data])
                labels.append(i[self.arg.d_data])

            for i in range(self.arg.d_data):
                for j in range(self.arg.d_data):
                    x[i][j] = np.float32(x[i][j])
            for i in range(len(labels)):
                labels[i] = np.float32(labels[i])
        x = np.array(x,dtype=float)
        labels = np.array(labels,dtype=float)
        return x, labels

    def split_anomaly(self, data, label):
        norm_tmp, anomaly_tmp = {'x': [], 'y': []}, {'x': [], 'y': []}
        for i in range(len(label)):
            if float(label[i]) == 0:  # 0 means norm data
                norm_tmp['x'].append(data[i])
                norm_tmp['y'].append(label[i])
            else:
                anomaly_tmp['x'].append(data[i])
                anomaly_tmp['y'].append(label[i])
        return norm_tmp, anomaly_tmp

    def shuffle(self, a, b):
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(a)
        random.seed(randnum)
        random.shuffle(b)
        return np.array(a), np.array(b)

    def split(self, norm_data, anomaly_data):
        train, test, select_anomaly, select_noise, select_train = {'x': [], 'y': []},{'x': [], 'y': []}, {'x': [], 'y': []}, {'x': [],'y': []}, {'x': [], 'y': []}
        norm_data['x'], norm_data['y'] = self.shuffle(norm_data['x'], norm_data['y'])
        anomaly_data['x'], anomaly_data['y'] = self.shuffle(anomaly_data['x'], anomaly_data['y'])

        length_norm = len(norm_data['y'])
        length_anomaly = len(anomaly_data['y'])
        batch_size_half = int(self.arg.batch_size / 2)
        batch_numbers = int(length_norm*0.8 / batch_size_half)
        anomaly_train_number = int(length_norm*0.8 * self.arg.radio)+1
        noise_num = int(0.02 * length_norm*0.8)

        select_anomaly['x'], select_anomaly['y'] = anomaly_data['x'][:anomaly_train_number], anomaly_data['y'][:anomaly_train_number]
        select_noise['x'], select_noise['y'] = anomaly_data['x'][anomaly_train_number:noise_num], norm_data['y'][anomaly_train_number:noise_num]
        # mixing
        select_train['x'] = np.concatenate((norm_data['x'][:int(0.8*length_norm)], select_noise['x']))
        select_train['y'] = np.concatenate((norm_data['y'][:int(0.8*length_norm)], select_noise['y']))
        select_train['x'], select_train['y'] = self.shuffle(select_train['x'], select_train['y'])
        print('noise_num:', noise_num)
        print('anomaly_num:',anomaly_train_number)
        num = 0
        for i in range(batch_numbers):
            for j in range(self.arg.batch_size):
                if j % 2 == 0:
                    train['x'].append(select_train['x'][num])
                    train['y'].append(select_train['y'][num])
                    num += 1
                else:
                    randnum = random.randint(0, anomaly_train_number-1)
                    train['x'].append(select_anomaly['x'][randnum])
                    train['y'].append(select_anomaly['y'][randnum])
        train['x'], train['y'] = np.array(train['x']), np.array(train['y'])
        train['x'], train['y'] = torch.from_numpy(train['x']), torch.from_numpy(train['y'])
        test['x'] = np.concatenate((norm_data['x'][int(0.8 * length_norm):], anomaly_data['x'][int(0.8 * length_anomaly):]))
        test['y'] = np.concatenate((norm_data['y'][int(0.8 * length_norm):], anomaly_data['y'][int(0.8 * length_anomaly):]))
        self.test_label = test['y']
        test['x'], test['y'] = torch.from_numpy(test['x']), torch.from_numpy(test['y'])
        self.test.append(DataLoader(dataset=TensorDataset(test['x'],test['y']),batch_size=self.arg.batch_size,shuffle=False))
        return train

    def spilt_into_client(self, norm_data, anomaly_data):
        length_norm_client = int(len(norm_data['y'])/self.arg.client)
        length_anomaly_client = int(len(anomaly_data['y'])/self.arg.client)
        for i in range(self.arg.client):
            norm = {'x': norm_data['x'][i*length_norm_client:(i+1)*length_norm_client],'y':norm_data['y'][i*length_norm_client:(i+1)*length_norm_client]}
            anomaly = {'x':anomaly_data['x'][i*length_anomaly_client:(i+1)*length_anomaly_client],'y':anomaly_data['y'][i*length_anomaly_client:(i+1)*length_anomaly_client]}
            self.client_dataset.append(self.split(norm, anomaly))

    def run(self):
        x, y = self.dataLoading()
        norm, anomaly = self.split_anomaly(x, y)
        self.spilt_into_client(norm,anomaly)
        for i in range(self.arg.client):
            self.client_dataset[i] = DataLoader(dataset=TensorDataset(self.client_dataset[i]['x'],self.client_dataset[i]['y']),batch_size=self.arg.batch_size,shuffle=False)
        return self.client_dataset
