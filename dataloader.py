from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')
        print(self.data1.shape)
        print(self.data2.shape)
        print(self.data3.shape)
        # scipy.io.savemat('CCV.mat', {'X1': self.data1, 'X2': self.data2, 'X3': self.data3, 'Y': self.labels})

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class Caltech_6V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)        
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
        # for i in [0, 3]:
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        # print(self.class_num)
        # for i in range(5000):
        #     print(data['X1'][i][-1])
        # X1 = data['X1'][:, :-1]
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class YoutubeVideo(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])

        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Caltech":
        dataset = Caltech_6V('data/Caltech.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "YoutubeVideo":
        dataset = YoutubeVideo("./data/Video-3V.mat", view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
