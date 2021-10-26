import time
import os
import sys

import random

import numpy as np
import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../")
from data_loader.libsvm_dataset import DenseLibsvmDataset2, DenseLibsvmDataset3
from data_loader.libsvm_dataset import DenseLibsvmDataset
from data_loader.libsvm_dataset import SparseLibsvmDataset


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_mnist(batch_size, path, download=True):
    """ Partitioning MNIST """
    train_dataset = datasets.MNIST(path, train=True, download=download,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307, ), (0.3081, ))]))
    test_dataset = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(dist.get_rank())
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, bsz, test_loader


# each worker only has several digits.
def partition_mnist_fl(path, n_workers, n_class_per_worker, download=True):
    n_classes = 10
    train_dataset = datasets.MNIST(path, train=True, download=download,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))

    print(train_dataset.data.shape)
    print(train_dataset.targets.shape)

    n_worker_per_class = int(n_class_per_worker * n_workers / n_classes)
    print("one class has data on {} workers".format(n_worker_per_class))

    count_classes = [0] * n_classes
    for i in range(len(train_dataset)):
        count_classes[train_dataset.targets[i]] += 1
    print("number of each class = {}".format(count_classes))

    class_place_count = {}
    for i in range(n_classes):
        class_place_count[i] = n_worker_per_class

    # class on each worker
    class_on_worker = np.zeros((n_workers, n_class_per_worker))
    class_on_worker.fill(-1)
    class_on_worker = np.asarray([[6.0, 9.0], [5.0, 3.0], [3.0, 9.0], [3.0, 8.0], [7.0, 0.0], [8.0, 1.0], [9.0, 6.0], [4.0, 9.0], [6.0, 3.0], [5.0, 0.0], [2.0, 5.0], [9.0, 1.0], [6.0, 0.0], [1.0, 3.0], [4.0, 9.0], [7.0, 8.0], [4.0, 7.0], [9.0, 5.0], [0.0, 4.0], [6.0, 4.0], [5.0, 1.0], [0.0, 8.0], [1.0, 2.0], [8.0, 9.0], [6.0, 5.0], [7.0, 1.0], [2.0, 7.0], [5.0, 7.0], [7.0, 2.0], [9.0, 4.0], [0.0, 9.0], [0.0, 2.0], [8.0, 3.0], [5.0, 6.0], [5.0, 6.0], [1.0, 2.0], [8.0, 5.0], [1.0, 3.0], [8.0, 7.0], [7.0, 2.0], [6.0, 2.0], [9.0, 0.0], [4.0, 0.0], [6.0, 1.0], [9.0, 2.0], [3.0, 9.0], [6.0, 9.0], [4.0, 3.0], [4.0, 7.0], [5.0, 4.0], [8.0, 2.0], [0.0, 8.0], [0.0, 8.0], [1.0, 9.0], [5.0, 1.0], [4.0, 3.0], [8.0, 1.0], [4.0, 2.0], [7.0, 0.0], [7.0, 0.0], [4.0, 5.0], [4.0, 2.0], [7.0, 3.0], [0.0, 6.0], [2.0, 1.0], [6.0, 5.0], [0.0, 4.0], [8.0, 7.0], [1.0, 9.0], [5.0, 6.0], [7.0, 3.0], [3.0, 7.0], [2.0, 9.0], [8.0, 6.0], [4.0, 7.0], [8.0, 9.0], [9.0, 5.0], [2.0, 3.0], [6.0, 9.0], [3.0, 5.0], [7.0, 3.0], [6.0, 2.0], [3.0, 6.0], [6.0, 3.0], [8.0, 7.0], [8.0, 0.0], [1.0, 2.0], [2.0, 3.0], [4.0, 1.0], [0.0, 6.0], [3.0, 5.0], [5.0, 1.0], [8.0, 7.0], [4.0, 8.0], [0.0, 2.0], [5.0, 0.0], [1.0, 8.0], [0.0, 2.0], [4.0, 1.0], [1.0, 4.0]])

    # workers of each class
    class_place = [[] for _ in range(n_classes)]
    class_place = [[4, 9, 12, 18, 21, 30, 31, 41, 42, 51, 52, 58, 59, 63, 66, 85, 89, 94, 95, 97], [5, 11, 13, 20, 22, 25, 35, 37, 43, 53, 54, 56, 64, 68, 86, 88, 91, 96, 98, 99], [10, 22, 26, 28, 31, 35, 39, 40, 44, 50, 57, 61, 64, 72, 77, 81, 86, 87, 94, 97], [1, 2, 3, 8, 13, 32, 37, 45, 47, 55, 62, 70, 71, 77, 79, 80, 82, 83, 87, 90], [7, 14, 16, 18, 19, 29, 42, 47, 48, 49, 55, 57, 60, 61, 66, 74, 88, 93, 98, 99], [1, 9, 10, 17, 20, 24, 27, 33, 34, 36, 49, 54, 60, 65, 69, 76, 79, 90, 91, 95], [0, 6, 8, 12, 19, 24, 33, 34, 40, 43, 46, 63, 65, 69, 73, 78, 81, 82, 83, 89], [4, 15, 16, 25, 26, 27, 28, 38, 39, 48, 58, 59, 62, 67, 70, 71, 74, 80, 84, 92], [3, 5, 15, 21, 23, 32, 36, 38, 50, 51, 52, 56, 67, 73, 75, 84, 85, 92, 93, 96], [0, 2, 6, 7, 11, 14, 17, 23, 29, 30, 41, 44, 45, 46, 53, 68, 72, 75, 76, 78]]

    # for i in range(n_workers):
    #     for j in range(n_class_per_worker):
    #         c = random.randint(0, n_classes - 1)
    #         while class_place_count[c] < 1 or np.any(class_on_worker[i, :] == c):
    #             c = random.randint(0, n_classes - 1)
    #         class_on_worker[i, j] = c
    #         class_place_count[c] = class_place_count[c] - 1
    #         class_place[c].append(i)

    for i in range(n_workers):
        unique_value = np.unique(class_on_worker[i, :])
        if unique_value.shape[0] < n_class_per_worker:
            print("worker {} has duplicate classes {}".format(i, class_on_worker[i]))

    print("classes on each worker = {}".format(class_on_worker.tolist()))
    print("place of each class = {}".format(class_place))

    for i in range(n_classes):
        print("place of class {} = {}, length = {}".format(i, class_place[i], len(class_place[i])))

    fl_train_data = []
    fl_train_target = []

    # row: worker, col: label
    count = np.zeros((n_workers, n_classes))

    class_place_index = [0] * n_classes

    # data for each worker
    for i in range(n_workers):
        fl_train_data.append([])
        fl_train_target.append([])

    # partition train dataset
    for i in range(len(train_dataset)):
        data, target = train_dataset.__getitem__(i)
        place_ind = class_place_index[target] % n_worker_per_class
        place_w_id = class_place[target][place_ind]
        fl_train_data[place_w_id].append(data)
        fl_train_target[place_w_id].append(target)
        count[place_w_id, target] += 1
        class_place_index[target] = (class_place_index[target] + 1) % n_worker_per_class

    for i in range(n_workers):
        data_file_name = "{}_{}_fl_data".format(i, n_workers)
        target_file_name = "{}_{}_fl_target".format(i, n_workers)
        train_data = np.stack(fl_train_data[i], axis=0)
        print("worker {} data shape = {}".format(i, train_data.shape))
        train_target = np.array(fl_train_target[i])
        unique_class = np.unique(train_target)
        if len(unique_class) != n_class_per_worker:
            print("worker {} has {} classes!!!".format(i, len(unique_class)))
        print("worker {} target shape = {} number of class = {}".format(i, train_target.shape, len(unique_class)))
        np.save(os.path.join(path, data_file_name), train_data)
        np.save(os.path.join(path, target_file_name), train_target)

    for i in range(n_workers):
        print("data distribution of worker {}: {}".format(i, count[i, :]))


def mnist_test_to_numpy(path, download=True):
    test_dataset = datasets.MNIST(path, train=False, download=download,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))

    fl_test_data = []
    fl_test_label = []

    for i in range(len(test_dataset.targets)):
        data, label = test_dataset.__getitem__(i)
        fl_test_data.append(data.reshape((28*28)))
        fl_test_label.append(label)

    return fl_test_data, fl_test_label


def load_mnist_fl(root_path, train_data_file, train_target_file, n_train_batch, test_batch_size):
    """ Load MNIST """

    start_time = time.time()
    train_data = np.load(os.path.join(root_path, train_data_file))
    print("read train data cost {} s".format(time.time() - start_time))
    n_train_data = train_data.shape[0]
    train_data = train_data.reshape((n_train_data, 28*28))
    print("train data shape = {}, dtype = {}".format(train_data.shape, train_data.dtype))
    #print("train data sample = {}".format(train_data[0]))

    train_target = np.load(os.path.join(root_path, train_target_file))
    print("read train target cost {} s".format(time.time() - start_time))
    print("train target shape = {}, dtype = {}".format(train_target.shape, train_target.dtype))
    unique_target = np.unique(train_target)
    print("this worker has {} unique classes: {}".format(len(unique_target), unique_target))
    print("train target sample = {}".format(train_target[0:10]))
    n_train_target = train_target.shape[0]

    if n_train_data != n_train_target:
        raise AssertionError("row of train data is {}, but row of label matrix is {}."
                             .format(n_train_data, n_train_target))

    parse_start = time.time()

    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()

    train_batch_size = 0
    if len(train_data) % n_train_batch == 0:
        train_batch_size = int(len(train_data) / n_train_batch)
    else:
        train_batch_size = int(len(train_data) / n_train_batch) + 1

    #train_dataset = CIFAR10_subset(True, list(data_matrix), list(labels_matrix), None, None)
    train_dataset = DenseLibsvmDataset3(train_data.shape[1], train_data, train_target)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    print("train batch size = {}, number of batches = {}".format(train_batch_size, len(train_loader)))

    # output is image
    # transform_test = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # test_dataset = datasets.CIFAR10(root_path, train=False, download=True, transform=transform_test)

    # output is list of numpy array
    test_data, test_target = mnist_test_to_numpy(root_path)
    print("test data length = {}, target length = {}".format(len(test_data), len(test_target)))
    #print("test data sample = {} : {}".format(test_target[0], test_data[0]))
    test_dataset = DenseLibsvmDataset3(len(test_data), np.stack(test_data, axis=0), np.asarray(test_target))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    print("test batch size = {}, number of batches = {}".format(test_batch_size, len(test_loader)))

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    print("preprocess data cost {} s, train dataset size = {}, test dataset size = {}"
          .format(time.time() - preprocess_start, train_dataset_size, test_dataset_size))

    return train_loader, test_loader


def partition_cifar10(batch_size, path, download=True):
    """ Partitioning Cifar10 """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(path, train=True, download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(path, train=False, download=download, transform=transform_test)
    size = 1
    rank = 0
    if dist_is_initialized():
        size = dist.get_world_size()
        rank = dist.get_rank()
    bsz = int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(rank)
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, bsz, test_loader


def generate_cifar10_fl(path, imbalance_ratio, download=True):
    """ Partitioning Cifar10 """
    n_classes = 10
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(path, train=True, download=download, transform=transform_train)

    print(train_dataset.data.shape)
    print(len(train_dataset.targets))

    np.random.seed(123)

    fl_train_data = []
    fl_train_label = []

    # row: worker, col: label
    count = np.zeros((10, 10))

    # data for 10 workers
    for i in range(n_classes):
        fl_train_data.append([])
        fl_train_label.append([])

    # partition train dataset
    for i in range(len(train_dataset.targets)):
        data, label = train_dataset.__getitem__(i)
        np_rand = np.random.uniform(0, 1)
        is_other = True if np_rand > imbalance_ratio else False
        if is_other:
            w_id = random.randint(0, n_classes-1)
            fl_train_data[w_id].append(data)
            fl_train_label[w_id].append(label)
            count[w_id, label] += 1
        else:
            fl_train_data[label].append(data)
            fl_train_label[label].append(label)
            count[label, label] += 1

    for i in range(n_classes):
        data_file_name = "{}_{}_fl_data".format(i, n_classes)
        label_file_name = "{}_{}_fl_label".format(i, n_classes)
        train_data = np.stack(fl_train_data[i], axis=0)
        print("worker {} data shape = {}".format(i, train_data.shape))
        train_label = np.array(fl_train_label[i])
        print("worker {} label shape = {}".format(i, train_label.shape))
        np.save(os.path.join(path, data_file_name), train_data)
        np.save(os.path.join(path, label_file_name), train_label)

    for i in range(n_classes):
        print("data distribution of worker {}: {}".format(i, count[i, :]))


def cifar10_test_to_numpy(root_path):
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.CIFAR10(root_path, train=False, download=True, transform=transform_test)

    fl_test_data = []
    fl_test_label = []

    for i in range(len(test_dataset.targets)):
        data, label = test_dataset.__getitem__(i)
        fl_test_data.append(data.reshape((32*32)))
        fl_test_label.append(label)

    return fl_test_data, fl_test_label


def load_cifar10_fl(root_path, features_path, target_path, batch_size):
    """ Load Cifar10 """

    start_time = time.time()
    train_data = np.load(features_path)
    print("read train data cost {} s".format(time.time() - start_time))
    print("train data shape = {}, dtype = {}".format(train_data.shape, train_data.dtype))
    n_train_data = train_data.shape[0]
    train_data = train_data.reshape((n_train_data, 32*32))
    print("train data shape = {}, dtype = {}".format(train_data.shape, train_data.dtype))
    print("train data sample = {}".format(train_data[0]))

    train_target = np.load(target_path)
    print("read train target cost {} s".format(time.time() - start_time))
    print("train target shape = {}, dtype = {}".format(train_target.shape, train_target.dtype))
    print("train target sample = {}".format(train_target[0:10]))
    n_train_target = train_target.shape[0]

    if n_train_data != n_train_target:
        raise AssertionError("row of train data is {}, but row of label matrix is {}."
                             .format(n_train_data, n_train_target))

    parse_start = time.time()

    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()

    #train_dataset = CIFAR10_subset(True, list(data_matrix), list(labels_matrix), None, None)
    train_dataset = DenseLibsvmDataset3(train_data.shape[1], train_data, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # output is image
    # transform_test = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # test_dataset = datasets.CIFAR10(root_path, train=False, download=True, transform=transform_test)

    # output is list of numpy array
    test_data, test_target = cifar10_test_to_numpy(root_path)
    print("test data length = {}, target length = {}".format(len(test_data), len(test_target)))
    print("test data sample = {} : {}".format(test_target[0], test_data[0]))
    test_dataset = DenseLibsvmDataset3(len(test_data), np.stack(test_data, axis=0), np.asarray(test_target))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    print("preprocess data cost {} s, train dataset size = {}, test dataset size = {}"
          .format(time.time() - preprocess_start, train_dataset_size, test_dataset_size))

    return train_loader, test_loader


def generate_vgg16_cifar100_fc_fl(path, data_file, target_file, imbalance_ratio):
    n_classes = 100

    start_time = time.time()
    data_matrix = np.load(os.path.join(path, data_file))
    print("read data matrix cost {} s".format(time.time() - start_time))
    print("data matrix shape = {}, dtype = {}".format(data_matrix.shape, data_matrix.dtype))
    print("data matrix sample = {}".format(data_matrix[0]))
    n_data = data_matrix.shape[0]
    col_data = data_matrix.shape[1]

    target_matrix = np.load(os.path.join(path, target_file))
    print("read target matrix cost {} s".format(time.time() - start_time))
    print("target matrix shape = {}, dtype = {}".format(target_matrix.shape, target_matrix.dtype))
    print("target matrix sample = {}".format(target_matrix[0:10]))
    n_target = target_matrix.shape[0]

    if n_data != n_target:
        raise AssertionError("row of data matrix is {}, but row of target matrix is {}."
                             .format(n_data, n_target))

    fl_train_data = []
    fl_train_label = []

    # row: worker, col: label
    count = np.zeros((n_classes, n_classes))

    # data for 100 workers
    for i in range(n_classes):
        fl_train_data.append([])
        fl_train_label.append([])

    # partition train dataset
    for i in range(n_data):
        data = data_matrix[i]
        label = target_matrix[i]
        np_rand = np.random.uniform(0, 1)
        is_other = True if np_rand > imbalance_ratio else False
        if is_other:
            w_id = random.randint(0, n_classes - 1)
            fl_train_data[w_id].append(data)
            fl_train_label[w_id].append(label)
            count[w_id, label] += 1
        else:
            fl_train_data[label].append(data)
            fl_train_label[label].append(label)
            count[label, label] += 1

    for i in range(n_classes):
        data_file_name = "{}_{}_fl_data".format(i, n_classes)
        target_file_name = "{}_{}_fl_target".format(i, n_classes)
        train_data = np.stack(fl_train_data[i], axis=0)
        print("worker {} data shape = {}".format(i, train_data.shape))
        train_label = np.array(fl_train_label[i])
        print("worker {} target shape = {}".format(i, train_label.shape))
        np.save(os.path.join(path, data_file_name), train_data)
        np.save(os.path.join(path, target_file_name), train_label)

    for i in range(n_classes):
        print("data distribution of worker {}: {}".format(i, count[i, :]))


def load_cifar10_fc_fl(root, train_data_file, train_target_file, test_data_file, test_target_file, batch_size):
    start_time = time.time()
    train_data = np.load(os.path.join(root, train_data_file))
    print("read train data cost {} s".format(time.time() - start_time))
    print("train data shape = {}, dtype = {}".format(train_data.shape, train_data.dtype))
    print("train data sample = {}".format(train_data[0]))
    n_train_data = train_data.shape[0]

    train_target = np.load(os.path.join(root, train_target_file))
    print("read train target cost {} s".format(time.time() - start_time))
    print("train target shape = {}, dtype = {}".format(train_target.shape, train_target.dtype))
    print("train target sample = {}".format(train_target[0:10]))
    n_train_target = train_target.shape[0]

    if n_train_data != n_train_target:
        raise AssertionError("row of train data is {}, but row of label matrix is {}."
                             .format(n_train_data, n_train_target))

    print("read data cost {} s".format(time.time() - start_time))

    preprocess_start = time.time()

    train_dataset = DenseLibsvmDataset3(train_data.shape[1], train_data, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = np.load(os.path.join(root, test_data_file))
    test_target = np.load(os.path.join(root, test_target_file))
    print("test data shape = {}, target shape = {}".format(test_data.shape, test_target.shape))
    print("test data sample = {} : {}".format(test_target[0], test_data[0]))
    test_dataset = DenseLibsvmDataset3(len(test_data), test_data, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    print("preprocess data cost {} s, train dataset size = {}, test dataset size = {}"
          .format(time.time() - preprocess_start, train_dataset_size, test_dataset_size))

    return train_loader, test_loader


def load_cifar100_fc_fl(root, train_data_file, train_target_file, test_data_file, test_target_file, batch_size):
    start_time = time.time()
    train_data = np.load(os.path.join(root, train_data_file))
    print("read train data cost {} s".format(time.time() - start_time))
    print("train data shape = {}, dtype = {}".format(train_data.shape, train_data.dtype))
    print("train data sample = {}".format(train_data[0]))
    n_train_data = train_data.shape[0]

    train_target = np.load(os.path.join(root, train_target_file))
    print("read train target cost {} s".format(time.time() - start_time))
    print("train target shape = {}, dtype = {}".format(train_target.shape, train_target.dtype))
    print("train target sample = {}".format(train_target[0:10]))
    n_train_target = train_target.shape[0]

    if n_train_data != n_train_target:
        raise AssertionError("row of train data is {}, but row of label matrix is {}."
                             .format(n_train_data, n_train_target))

    print("read data cost {} s".format(time.time() - start_time))

    preprocess_start = time.time()

    train_dataset = DenseLibsvmDataset3(train_data.shape[1], train_data, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = np.load(os.path.join(root, test_data_file))
    test_target = np.load(os.path.join(root, test_target_file))
    print("test data shape = {}, target shape = {}".format(test_data.shape, test_target.shape))
    print("test data sample = {} : {}".format(test_target[0], test_data[0]))
    test_dataset = DenseLibsvmDataset3(len(test_data), test_data, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    print("preprocess data cost {} s, train dataset size = {}, test dataset size = {}"
          .format(time.time() - preprocess_start, train_dataset_size, test_dataset_size))

    return train_loader, test_loader


def partition_vgg16_cifar100_fc(batch_size, features_path, labels_path, validation_ratio, shuffle=True):
    """ Partitioning fully-connected layer of vgg16 on cifar100"""

    start_time = time.time()
    features_matrix = np.load(features_path)
    print("read features matrix cost {} s".format(time.time() - start_time))
    print("feature matrix shape = {}, dtype = {}".format(features_matrix.shape, features_matrix.dtype))
    print("feature matrix sample = {}".format(features_matrix[0]))
    row_features = features_matrix.shape[0]
    col_features = features_matrix.shape[1]

    labels_matrix = np.load(labels_path)
    print("read label matrix cost {} s".format(time.time() - start_time))
    print("label matrix shape = {}, dtype = {}".format(labels_matrix.shape, labels_matrix.dtype))
    print("label matrix sample = {}".format(labels_matrix[0:10]))
    row_labels = labels_matrix.shape[0]

    if row_features != row_labels:
        raise AssertionError("row of feature matrix is {}, but row of label matrix is {}."
                             .format(row_features, row_labels))

    parse_start = time.time()
    dataset = DenseLibsvmDataset3(col_features, features_matrix, labels_matrix)
    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    print("preprocess data cost {} s, dataset size = {}"
          .format(time.time() - preprocess_start, dataset_size))

    return train_loader, validation_loader


def parse_libsvm(line, max_features):
    splits = line.split()
    label = int(splits[0])
    values = np.zeros(max_features, dtype=np.float32)
    for item in splits[1:]:
        tup = item.split(":")
        values[int(tup[0]) - 1] = float(tup[1])
    return label, values


def vertical_partition_higgs_with_label(in_file_name, dst_path, n_features, n_part):
    print("vertically partition {} into {} splits".format(in_file_name, n_part))

    n_features_part = 0
    if n_features % n_part == 0:
        n_features_part = int(n_features / n_part)
    else:
        n_features_part = int(n_features / n_part) + 1
    print("each split has {} features".format(n_features_part))

    in_file = open(in_file_name)

    out_files = []
    for i in np.arange(n_part):
        file_name = "{}_{}".format(i, n_part)
        dst_file_name = os.path.join(dst_path, file_name)
        out_files.append(open(dst_file_name, "w"))

    for line in in_file:
        if line is not None:
            label, features = parse_libsvm(line, n_features)
            for i in np.arange(n_part):
                f_start = i * n_features_part
                f_end = min(n_features, (i+1) * n_features_part)
                line_part = features[f_start:f_end]
                out_files[i].write(",".join(map(str, line_part)))
                out_files[i].write("," + str(label) + "\n")

    for f in out_files:
        f.close()


def load_dummy_partition_with_label(file_name):
    f = open(file_name)
    data = []
    label = []

    for line in f:
        parsed = np.fromstring(line, dtype=float, sep=",")
        data.append(parsed[:-1])
        label.append(int(parsed[-1:]))

    f.close()
    return np.asarray(data), np.asarray(label)


def partition_higgs(batch_size, file_name, validation_ratio):
    parse_start = time.time()
    f = open(file_name).readlines()
    dataset = DenseLibsvmDataset2(f, 30)
    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=valid_sampler)

    print("preprocess data cost {} s".format(time.time() - preprocess_start))
    return train_loader, test_loader


def partition_yfcc100m(file_list, n_features, pos_tag, batch_size, validation_ratio):
    parse_start = time.time()
    f = open(file_list[0]).readlines()
    dataset = DenseLibsvmDataset(f, n_features, pos_tag)
    if len(file_list) > 1:
        for file_name in file_list[1:]:
            f = open(file_name).readlines()
            dataset.add_more(f)

    total_count = dataset.__len__()
    pos_count = 0
    for i in range(total_count):
        if dataset.__getitem__(i)[1] == 1:
            pos_count += 1
    print("{} positive observations out of {}".format(pos_count, total_count))

    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=valid_sampler)

    print("preprocess data cost {} s".format(time.time() - preprocess_start))
    return train_loader, test_loader


def partition_agaricus(batch_size, train_file, test_file):
    train_dataset = SparseLibsvmDataset(train_file, 127)
    test_dataset = SparseLibsvmDataset(test_file, 127)

    size = dist.get_world_size()
    bsz = 1 if batch_size == 1 else int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(dist.get_rank())
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_partition, train_loader, bsz, test_loader


def partition_sparse(file, num_feature):
    train_dataset = SparseLibsvmDataset(file, num_feature)
    size = 1
    rank = 0
    if dist_is_initialized():
        size = dist.get_world_size()
        rank = dist.get_rank()
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(rank)
    return train_partition


if __name__ == '__main__':
    path = "D:\\Downloads\\datasets\\cifar10\\"
    #generate_cifar10_fl(path, 0.8)
    # feature_file = path + "1_10_fl_data.npy"
    # label_file = path + "1_10_fl_label.npy"
    # load_cifar10_fl(path, feature_file, label_file, 500)

    # generate cifar100 fc
    # path = "D:\\Downloads\\datasets\\cifar100_fc_FL\\"
    # data_file = "data_vgg19_train.npy"
    # target_file = "target_vgg19_train.npy"
    # generate_vgg16_cifar100_fc_fl(path, data_file, target_file, 0.1)

    # load cifar100 fc
    # train_data_file = "0_100_fl_data.npy"
    # train_target_file = "0_100_fl_target.npy"
    # test_data_file = "data_vgg19_test.npy"
    # test_target_file = "target_vgg19_test.npy"
    # load_cifar100_fc_fl(path, train_data_file, train_target_file, test_data_file, test_target_file, 50)

    path = "D:\\Downloads\\datasets\\mnist\\"
    #partition_mnist_fl(path, 100, 2, download=False)
    train_data = "1_100_fl_data.npy"
    train_target = "1_100_fl_target.npy"
    load_mnist_fl(path, train_data, train_target, 20, 100)
