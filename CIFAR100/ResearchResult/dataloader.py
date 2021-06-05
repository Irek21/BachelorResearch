import numpy as np
import torch
import pickle as pkl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchLoader():
    def __init__(self, class_size, num_classes, batch_size, shuffle, RAM,
                 data=None, labels=None, batches_in_buff=None, path=None, first_class=None):
        self.class_size = class_size
        self.num_classes = num_classes
        self.train_size = num_classes * class_size
        self.batch_size = batch_size
        self.indices = np.random.permutation(num_classes * class_size) if shuffle else np.arange(num_classes * class_size)
        self.RAM = RAM
        if RAM:
            self.data = data[self.indices] if shuffle else data
            self.labels = labels[self.indices] if shuffle else labels
        else:
            self.batches_in_buff = batches_in_buff
            self.buff_size = batches_in_buff * batch_size
            self.buff = [{'label': 0, 'features': torch.zeros(2048, device=device)} for i in range(self.buff_size)]
            self.buff_num = 0
            self.first_class = first_class
            self.path = path
    
    def buff_gen(self, buff_num):
        buff_indices = self.indices[buff_num * self.buff_size:(buff_num + 1) * self.buff_size]

        for i in range(self.num_classes):
            with open(self.path + str(self.first_class + i), 'rb') as f:
                class_data = pkl.load(f)

            class_indices = np.where(((buff_indices < (i + 1) * self.class_size) & (buff_indices >= i * self.class_size)))[0]
            for j in class_indices:
                self.buff[j] = {
                    'label': class_data['label'],
                    'features': class_data['features'][buff_indices[j] % self.class_size]
                }
    
    def batch_load(self, i):
        if self.RAM:
            return self.data[i * self.batch_size:(i + 1) * self.batch_size], self.labels[i * self.batch_size:(i + 1) * self.batch_size]
        else:
            buff_i = i % self.batches_in_buff
            if (buff_i == 0):
                self.buff_gen(self.buff_num)
                self.buff_num += 1

            return self.buff[buff_i * self.batch_size:(buff_i + 1) * self.batch_size]