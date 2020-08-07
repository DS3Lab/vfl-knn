import numpy as np


def square_euclidean(dataset, pred_data):
    dist_arr = np.zeros(len(dataset), dtype=np.float32)
    for i in np.arange(len(dataset)):
        data = dataset[i]
        square_dist = np.sum((data-pred_data)**2)
        dist_arr[i] = square_dist
    return dist_arr


def square_euclidean_np(dataset, pred_data):
    dist = np.sum((dataset - pred_data)**2, axis=1)
    return dist


def largest_indices(array: np.ndarray, n: int):
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    return indices


def least_indices(array: np.ndarray, n: int):
    flat = array.flatten()
    indices = np.argpartition(flat, n)[:n]
    print(array)
    return indices


if __name__ == "__main__":
    a = np.asarray([[0,1,2,3,4], [5,6,7,8,9]])
    b = np.asarray([0,1,2,3,4])
    #print(square_euclidean_np(a, b))
    c = np.asarray([4,1,6,2,8,3,0,5])
    ind = least_indices(c, 3)
    print(ind)
