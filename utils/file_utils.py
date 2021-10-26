import os

import numpy as np


def parse_libsvm(line, max_features):
    splits = line.split()
    label = int(splits[0])
    values = np.zeros(max_features, dtype=np.float32)
    for item in splits[1:]:
        tup = item.split(":")
        values[int(tup[0]) - 1] = float(tup[1])
    return label, values


def parse_csv(line, max_features):
    splits = line.split(",")
    label = int(splits[-1])
    values = np.zeros(max_features, dtype=np.float32)
    for i in range(len(splits) - 1):
        item = splits[i]
        values[i] = float(item)
    return label, values


def vertical_partition_libsvm(in_file_name, dst_path, n_features, n_part):
    print("vertically partition {} into {} splits".format(in_file_name, n_part))

    n_features_part = 0

    if n_features % n_part == 0:
        n_features_part = int(n_features / n_part)
    else:
        n_features_part = int((n_features - n_features % n_part) / (n_part - 1))

    print("each split has {} features".format(n_features_part))

    in_file = open(in_file_name)

    out_files = []
    for i in np.arange(n_part):
        file_name = "{}_{}".format(i, n_part)
        dst_file_name = os.path.join(dst_path, file_name)
        out_files.append(open(dst_file_name, "w"))

    for line in in_file:
        if line is not None:
            _, features = parse_libsvm(line, n_features)
            for i in np.arange(n_part):
                f_start = i * n_features_part
                f_end = min(n_features, (i+1) * n_features_part)
                line_part = features[f_start:f_end]
                out_files[i].write(",".join(map(str, line_part)) + "\n")

    for f in out_files:
        f.close()


def vertical_partition_libsvm_with_label(in_file_name, dst_path, n_features, n_part):
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
            if label == -1:
                label = 0
            for i in np.arange(n_part):
                f_start = i * n_features_part
                f_end = min(n_features, (i+1) * n_features_part)
                line_part = features[f_start:f_end]
                out_files[i].write(",".join(map(str, line_part)))
                out_files[i].write("," + str(label) + "\n")

    for f in out_files:
        f.close()


def vertical_partition_csv_with_label(in_file_name, dst_path, n_features, n_part):
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

    n_line = 0
    for line in in_file:
        if line is not None and n_line > 0:
            label, features = parse_csv(line, n_features)
            if label == -1:
                label = 0
            for i in np.arange(n_part):
                f_start = i * n_features_part
                f_end = min(n_features, (i+1) * n_features_part)
                line_part = features[f_start:f_end]
                out_files[i].write(",".join(map(str, line_part)))
                out_files[i].write("," + str(label) + "\n")
        n_line += 1

    for f in out_files:
        f.close()


def parse_part_line(line):
    return np.fromstring(line, dtype=float, sep=",")


def load_part(file_name):
    f = open(file_name)
    data = []

    for line in f:
        data.append(parse_part_line(line))

    f.close()
    return np.asarray(data)


def load_part_with_label(file_name):
    f = open(file_name)
    data = []
    label = []

    for line in f:
        parsed = parse_part_line(line)
        data.append(parsed[:-1])
        label.append(int(parsed[-1:]))

    f.close()
    return np.asarray(data), np.asarray(label)


if __name__ == "__main__":
    # file_name = "cod-rna.txt"
    # out_dir = "cod-rna-vertical"
    # vertical_partition_libsvm_with_label(file_name, out_dir, 8, 4)

    file_name = "unbalance-all.csv"
    out_dir = "unbalance-vertical"
    vertical_partition_csv_with_label(file_name, out_dir, 2, 2)
