import numpy as np


def sum_all_combinations(encrypted_list):
    n_clients = len(encrypted_list)
    combination_keys = generate_all_combinations(n_clients)
    n_combinations = len(combination_keys)
    print("there are {} combinations for {} clients: {}"
          .format(n_combinations, n_clients, combination_keys))

    # init list = size-1 combination
    summed_list = encrypted_list.copy()
    # key -> index
    key_dict = { combination_keys[i]: i for i in range(n_combinations) }

    for i in range(n_clients, n_combinations):
        key = combination_keys[i]
        key_index = key_dict[key]
        pre_index = key.rfind("_")
        if pre_index != -1:
            pre_key = key[:pre_index]
            post_key = key[pre_index + 1:]
            pre_key_index = key_dict[pre_key]
            post_key_index = key_dict[post_key]
            print("key {}, pre key {}, pre key index {}, post key {}, post key index {}"
                  .format(key, pre_key, pre_key_index, post_key, post_key_index))
            #print(summed_list)
            if type(summed_list[pre_key_index]) == list:
                summed_list.append(list(
                    map(sum, zip(summed_list[pre_key_index], summed_list[post_key_index]))
                ))
            else:
                summed_list.append(summed_list[pre_key_index] + summed_list[post_key_index])
        else:
            print("cannot find _ in key {}".format(key))
    return summed_list


def generate_all_combinations(length):
    ret = []
    if length >= 1:
        ret.extend(combination_1(length))
    if length >= 2:
        ret.extend(combination_2(length))
    if length >= 3:
        ret.extend(combination_3(length))
    if length >= 4:
        ret.extend(combination_4(length))
    if length >= 5:
        ret.extend(combination_5(length))
    if length >= 6:
        ret.extend(combination_6(length))
    if length >= 7:
        ret.extend(combination_7(length))
    if length >= 8:
        ret.extend(combination_8(length))
    if length >= 9:
        ret.extend(combination_9(length))
    return ret


def combination_1(length):
    return [str(i) for i in range(length)]


def combination_2(length):
    assert length >= 2
    ret = []
    for a in range(length - 1):
        for b in range(a + 1, length):
            ret.append("_".join(str(x) for x in [a, b]))
    return ret


def combination_3(length):
    assert length >= 3
    ret = []
    for a in range(length - 2):
        for b in range(a + 1, length - 1):
            for c in range(b + 1, length):
                ret.append("_".join(str(x) for x in [a, b, c]))
    return ret


def combination_4(length):
    size = 4
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    ret.append("_".join(str(x) for x in [a, b, c, d]))
    return ret


def combination_5(length):
    size = 5
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    for e in range(d + 1, length - size + 5):
                        ret.append("_".join(str(x) for x in [a, b, c, d, e]))
    return ret


def combination_6(length):
    size = 6
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    for e in range(d + 1, length - size + 5):
                        for f in range(e + 1, length - size + 6):
                            ret.append("_".join(str(x) for x in [a, b, c, d, e, f]))
    return ret


def combination_7(length):
    size = 7
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    for e in range(d + 1, length - size + 5):
                        for f in range(e + 1, length - size + 6):
                            for g in range(f + 1, length - size + 7):
                                ret.append("_".join(str(x) for x in [a, b, c, d, e, f, g]))
    return ret


def combination_8(length):
    size = 8
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    for e in range(d + 1, length - size + 5):
                        for f in range(e + 1, length - size + 6):
                            for g in range(f + 1, length - size + 7):
                                for h in range(g + 1, length - size + 8):
                                    ret.append("_".join(str(x) for x in [a, b, c, d, e, f, g, h]))
    return ret


def combination_9(length):
    size = 9
    assert length >= size
    ret = []
    for a in range(length - size + 1):
        for b in range(a + 1, length - size + 2):
            for c in range(b + 1, length - size + 3):
                for d in range(c + 1, length - size + 4):
                    for e in range(d + 1, length - size + 5):
                        for f in range(e + 1, length - size + 6):
                            for g in range(f + 1, length - size + 7):
                                for h in range(g + 1, length - size + 8):
                                    for i in range(h + 1, length - size + 9):
                                        ret.append("_".join(str(x) for x in [a, b, c, d, e, f, g, h, i]))
    return ret


if __name__ == '__main__':
    arr = [[1, 2, 3], [3, 2, 1], [1, 1, 1]]
    summed_list = []
    summed_list.extend(sum_all_combinations(arr))
    print(summed_list)

    print(len(arr))
    client_combinations = generate_all_combinations(len(arr))
    print(client_combinations)
    k = 3

    # find top-k for each summed list
    for i in range(len(summed_list)):
        cur_sum_list = summed_list[i]
        local_dist_ind = np.argsort(np.array(cur_sum_list))
        print("top-k items in group {}: {}"
              .format(client_combinations[i], local_dist_ind[:k]))
