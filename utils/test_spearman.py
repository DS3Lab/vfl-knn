from scipy import stats


def rank_to_value(arr):
    ret = [0] * len(arr)
    for i in range(len(arr)):
        rank = i
        ind = arr[i]
        ret[ind] = rank
    return ret


if __name__ == "__main__":
    rank1 = [2, 1, 0, 3]
    rank1_value = rank_to_value(rank1)
    print(rank1_value)

    rank2 = [2, 0, 1, 3]
    rank2_value = rank_to_value(rank2)
    print(rank2_value)

    print(stats.spearmanr(rank1_value, rank2_value))
