import keras
import numpy as np
import os
import cv2


class ShotDataset(keras.utils.Sequence):
    def __init__(self, train_list_path, data_dir, resize_shape=(112, 112), batch_size=1):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.resize_shape = resize_shape
        with open(train_list_path, 'r') as train_list_f:
            self.train_strings = [i.rstrip().split("DeepSBD/")[-1] for i in train_list_f.readlines()]
            np.random.shuffle(self.train_strings)
        self.idxs = list(range(len(self.train_strings)))
        np.random.shuffle(self.idxs)

    def __len__(self):
        return len(self.train_strings) // self.batch_size


    def __data_generation(self, idxs):

        X = []
        Y = []
        Y = np.zeros((len(idxs), 3), dtype=np.float32)
        for k, idx in enumerate(idxs):
            path, _, label = self.train_strings[idx].split()
            label = int(label)
            pics = [i for i in os.listdir(self.data_dir + '/' + path) if i.endswith('.jpg')]
            seq = []
            for pic_name in pics:
                img = cv2.imread(self.data_dir + '/' + path + '/' + pic_name, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.resize_shape)

                seq.append((img - 128) / 255)
            seq = np.array(seq)
            seq = seq[np.argsort(pics)]
            seq = seq.reshape(16, -1)
            seq = seq.astype(np.float32)
            X.append(seq)
            Y[k][label] = 1

        X = np.array(X)
        return X, Y


    def __getitem__(self, idx):

        idxs = self.idxs[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(idxs)
        return X, y
