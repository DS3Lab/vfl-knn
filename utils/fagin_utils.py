import numpy as np
import random

from collections import Counter


def count_by_dict(lists, count_map, cur_top_k, n_k):
    n_list = len(lists)
    n_item = lists[0].shape[0]
    print("number of list = {}, number of items = {}".format(n_list, n_item))
    for i in range(n_item):
        for j in range(n_list):
            nid = lists[j][i]
            cur_count = count_map.get(nid, 0)
            if cur_count == n_list - 1:
                cur_top_k.append(nid)
                if len(cur_top_k) == n_k:
                    return
            else:
                count_map[nid] = cur_count + 1
    return


def count_by_arr_old(lists, count_arr, cur_top_k, n_k):
    n_list = len(lists)
    n_item = lists[0].shape[0]
    #print("number of list = {}, number of items = {}".format(n_list, n_item))
    for i in range(n_item):
        for j in range(n_list):
            nid = lists[j][i]
            cur_count = count_arr[nid]
            if cur_count == n_list - 1:
                cur_top_k.append(nid)
                if len(cur_top_k) == n_k:
                    return
            else:
                count_arr[nid] = cur_count + 1
    return


def count_by_arr(lists, count_arr, cur_top_k, n_k):
    #problem with n_list
    n_list = len(lists) - 1 # We remove list from server
    n_item = lists[0].shape[0]

    #print("number of list n_list = {}, number of items n_item = {}".format(n_list, n_item))
    for i in range(n_item):
        for j in range(1,n_list+1): # we start at one so we do not take into consideration list from rank 0
                                    # Also we go to n+1 since we need to check last list
            #Problem here, if we keep encrypted ind, it will be difficult to compare
            nid = lists[j][i]
            #We try to access a value by passing an encrypted valie
            cur_count = count_arr[nid]
            #print("for nid: {} we have count: {}".format(nid,cur_count))
            if cur_count == n_list - 1:
                cur_top_k.append(nid)
                if len(cur_top_k) == n_k:
                    return
            else:
                # Even with Dict we cannot to that because encryption is not bijective
                #ATTENTION: we need to know ID in order to increase count
                # But to clients we will only send top-k ids and no more. Can it work?
                count_arr[nid] = cur_count + 1
    return


def count_by_np(np_arr, count_arr, cur_top_k, n_k):
    n_list = np_arr.shape[0]
    n_item = np_arr.shape[1]
    print("number of list = {}, number of items = {}".format(n_list, n_item))
    unique_elements, counts_elements = np.unique(np_arr.flatten(), return_counts=True)
    print("unique elements = {}".format(unique_elements[:10]))
    print("counts elements = {}".format(counts_elements[:10]))
    for i in range(len(unique_elements)):
        nid = unique_elements[i]
        count = counts_elements[i]
        cur_count = count_arr[nid]
        if cur_count == n_list - count:
            cur_top_k.append(nid)
            if len(cur_top_k) == n_k:
                return
        else:
            count_arr[nid] = cur_count + count
    return


def count_by_arr_kmeans(lists, count_arr, cur_top_k, n_k):
    n_list = len(lists)
    len_list = [list.shape[0] for list in lists]
    max_item = max(len_list)
    print("number of list = {}, number of items = {}".format(n_list, len_list))
    for i in range(max_item):
        for j in range(n_list):
            if i < len_list[j]:
                nid = lists[j][i]
                cur_count = count_arr[nid]
                if cur_count == n_list - 1:
                    cur_top_k.append(nid)
                    if len(cur_top_k) == n_k:
                        return
                else:
                    count_arr[nid] = cur_count + 1
    return

def createLookUpTable(dataSize, seed):
    random.seed(seed)
    return np.array(random.sample(range(dataSize), dataSize))

def get_shuffled_ind(ind, lookUpTable):
    return np.array(lookUpTable[ind])

def find(target, myList):
    for i in range(len(myList)):
        if myList[i] == target:
            return i
def get_real_ind(shuffled, lookUpTable):
    return list(map(lambda x: find(x, lookUpTable), shuffled))

def count_lists3(lists):
    flat_lists = np.asarray(lists).flat
    new_counter = Counter(flat_lists)
    return new_counter


def suggest_size(n_data, k, n_list):
    return int(n_data * pow(k / float(n_data), 1 / float(n_list)))


if __name__ == "__main__":
    arr = np.asarray([1,2,3,4,5])
    print(np.where(arr == 1))
