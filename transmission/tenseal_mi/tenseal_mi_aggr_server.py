import time
from concurrent import futures
from multiprocessing import Process, Queue

import numpy as np
import torch
import grpc
import tenseal as ts

import sys
sys.path.append("../../")
from transmission.tenseal_mi import tenseal_mi_aggr_server_pb2, tenseal_mi_aggr_server_pb2_grpc
from transmission.tenseal_mi.tenseal_mi_sche_client import MIScheClient


def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key // 2
    return client_attendance


class MIAggrServer(tenseal_mi_aggr_server_pb2_grpc.MIAggrServerServiceServicer):

    def __init__(self, aggr_address, sche_addresses, num_clients, k, ctx_file):
        self.aggr_address = aggr_address
        self.sche_addresses = sche_addresses.split(",")
        self.num_clients = num_clients
        # each node runs a worker and a schedule server
        assert len(self.sche_addresses) == self.num_clients
        self.schedule_clients = [MIScheClient(address, k, ctx_file)
                                 for address in self.sche_addresses]
        self.k = k
        self.num_candidate = 0

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.sleep_time = 0.01

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.client_vectors = [torch.tensor(0) for _ in range(num_clients)]
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

        self.group_keys = []
        self.group_sorted_ids = []

        print("mi aggregation server has been initialized")

    def reset_sum(self):
        self.num_candidate = 0
        self.client_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False
        self.group_keys = []
        self.group_sorted_ids = []

    def transmit(self, q, i, group_key):
        print("send to sched server: address: {}, group index {}, group key = {}"
              .format(self.sche_addresses, i, group_key))
        schedule_ind = i % len(self.schedule_clients)
        tmp = self.schedule_clients[schedule_ind].transmit(group_key, self.num_candidate, self.sum_data[i])
        # add group index for check
        tmp.append(i)
        q.put(tmp)
        return

    def multi_thread_trans(self):
        q = Queue()
        processes = []
        rets = []

        assert len(self.sum_data) == len(self.group_keys)

        for i in range(len(self.group_keys)):
            t = Process(target=self.transmit, args=(q, i, self.group_keys[i]))
            processes.append(t)
        for p in processes:
            p.start()
        for i in range(len(processes)):
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()

        for group_ind in range(len(self.group_keys)):
            for elem in rets:
                if elem[-1] == group_ind:
                    self.group_sorted_ids.extend((elem[:-1]))

        print("sched server return top-{}, size {}".format(self.num_candidate, len(self.group_sorted_ids)))
        return self.group_sorted_ids

    def sum_all_combinations(self):
        assert len(self.client_vectors) == self.num_clients
        n_groups = len(self.group_keys)

        summed_vecs = []
        for group_ind in range(n_groups):
            group_key = self.group_keys[group_ind]
            group_flags = utility_key_to_groups(group_key, self.num_clients)
            first_ind = group_flags.index(1)
            cur_sum_vec = self.client_vectors[first_ind]
            for i in range(first_ind + 1, len(group_flags)):
                if group_flags[i] == 1:
                    if type(self.client_vectors[i]) == list:
                        cur_sum_vec = [x + y for (x, y) in zip(cur_sum_vec, self.client_vectors[i])]
                    else:
                        cur_sum_vec += self.client_vectors[i]
            summed_vecs.append(cur_sum_vec)

        return summed_vecs

    def aggr_mi(self, request, context):

        server_start = time.time()

        client_rank = request.client_rank
        n_candidate = request.k
        # the leader client, 0 is the fagin server
        if client_rank == 1:
            self.num_candidate = request.k
            self.group_keys = list(request.groups)

        print(">>> server receive encrypted data from client {}, num candidate ={}, time = {} ----"
              .format(client_rank, n_candidate, time.asctime(time.localtime(time.time()))))

        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start

        # add received data to cache
        self.client_vectors[client_rank - 1] = enc_vector
        self.n_sum_request += 1

        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start

        # sum encrypted vector
        sum_time = 0
        sche_server_time = 0
        if client_rank == 1:
            sum_start = time.time()
            fagin_summed_vector = self.sum_all_combinations()
            self.sum_data.extend(fagin_summed_vector)
            sum_time = time.time() - sum_start

            # send to schedule servers to decrypt and sort
            sche_server_start = time.time()
            self.multi_thread_trans()
            sche_server_time = time.time() - sche_server_start

            self.sum_completed = True

        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start

        # create response
        response_start = time.time()
        response = tenseal_mi_aggr_server_pb2.mi_aggr_top_k(
            client_rank=client_rank,
            ranking=self.group_sorted_ids
        )
        response_time = time.time() - response_start

        # wait until creating all response
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == 1:
            self.reset_sum()

        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        print(">>> aggr server cost {:.2f} s: deserialization {:.2f} s, "
              "wait for requests {:.2f} s, sum {:.2f} s, comm with schedule servers  {:.2f} s, "
              "wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time,
                      wait_time, sum_time, sche_server_time, sum_wait_time, response_time))

        return response


def launch_server(aggr_address, sche_addresses, num_clients, k, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = MIAggrServer(aggr_address, sche_addresses, num_clients, k, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_mi_aggr_server_pb2_grpc.add_MIAggrServerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(aggr_address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    aggr_address = args[0]
    # first char is ","
    sche_addresses = args[1][1:]
    num_clients = int(args[2])
    k = int(args[3])
    ctx_file = args[4]
    launch_server(aggr_address, sche_addresses, num_clients, k, ctx_file)
