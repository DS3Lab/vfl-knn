import time
import os

import numpy as np
import argparse

import torch
import torch.distributed as dist

import crypten
import crypten.communicator as comm


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["distributed_backend".upper()] = str(args.backend)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)
    os.environ["RENDEZVOUS"] = "env://"
    os.environ["MASTER_ADDR"] = "bach03"
    os.environ["MASTER_PORT"] = "24000"

    crypten.init()

    test_start = time.time()

    for i in range(args.n_test):
        print(">>>>>> test[{}] <<<<<<".format(i))

        one_test_start = time.time()

        local_tensor = torch.tensor([i for i in range(args.tensor_size)])
        enc_tensor = crypten.cryptensor(local_tensor)

        sum_enc = comm.get().all_reduce(enc_tensor, op=dist.ReduceOp.SUM)

        sum_dec = sum_enc.get_plain_text()

        print(sum_dec[:10])

        one_test_time = time.time() - one_test_start

        print("one test finish, cost {} s".format(one_test_time))

    print("======finish======")
    print("test {} data cost {} s".format(args.n_test, time.time() - test_start))


N_CLIENTS = 5
N_TEST = 100
TENSOR_SIZE = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=N_CLIENTS, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--n-test', type=int, default=N_TEST)
    parser.add_argument('--tensor-size', type=int, default=TENSOR_SIZE)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    # if args.world_size > 1:
    #     dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
