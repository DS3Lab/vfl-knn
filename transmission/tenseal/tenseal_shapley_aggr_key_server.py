import time
from concurrent import futures
from multiprocessing import Process, Queue

import numpy as np
import grpc
import tenseal as ts

import sys
sys.path.append("../../")
from transmission.tenseal import tenseal_shapley_data_pb2, tenseal_shapley_data_pb2_grpc
from utils.shapley_utils import generate_all_combinations, sum_all_combinations

from transmission.tenseal.tenseal_key_server_client import KeyServerClient


class ShapleyAggrKeyServer(tenseal_shapley_data_pb2_grpc.ShapleyServiceServicer):

    def __init__(self, address, key_address, num_clients, k, ctx_file):
        self.address = address
        self.key_address = key_address
        self.num_clients = num_clients

        self.key_server_client = KeyServerClient(self.key_address, k, ctx_file)
        self.client_threads = 4
        self.k = k

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.sleep_time = 0.01

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.sum_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

        self.group_top_k = []

        print("shapley schedule server has been initialized")

    def reset_sum(self):
        self.sum_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False
        self.group_top_k = {}

    def transmit(self, q, i, group_key):
        print("send to key server: address: {}, group index {}, group key = {}"
              .format(self.key_address, i, group_key))
        tmp = self.key_server_client.transmit(group_key, self.sum_data[i])
        # add group index for check
        tmp.append(i)
        q.put(tmp)
        return

    def multi_thread_trans(self):
        q = Queue()
        processes = []
        rets = []

        client_combinations = generate_all_combinations(self.num_clients)

        for i in range(len(self.sum_data)):
            t = Process(target=self.transmit, args=(q, i, client_combinations[i]))
            processes.append(t)
        for p in processes:
            p.start()
        for i in range(len(processes)):
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()

        for group_ind in range(len(self.sum_data)):
            for elem in rets:
                if elem[-1] == group_ind:
                    self.group_top_k.extend((elem[:-1]))

        print("key server return shapley top-k, size {}".format(len(self.group_top_k)))
        return self.group_top_k

    def sum_shapley(self, request, context):

        server_start = time.time()

        client_rank = request.client_rank
        k = request.k

        print(">>> server receive encrypted data from client {}, k ={}, time = {} ----"
              .format(client_rank, k, time.asctime(time.localtime(time.time()))))

        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start

        # add received data to cache
        self.sum_vectors.append(enc_vector)
        self.n_sum_request += 1

        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start

        # sum encrypted vector
        sum_time = 0
        if client_rank == self.num_clients - 1:
            sum_start = time.time()
            fagin_summed_vector = sum_all_combinations(self.sum_vectors)
            self.sum_data.extend(fagin_summed_vector)

            # find top-k for each summed list
            self.multi_thread_trans()

            # client_combinations = generate_all_combinations(self.num_clients)
            # for i in range(len(self.sum_data)):
            #     cur_sum_list = self.sum_data[i]
            #     top_k_ind = self.key_server_client.transmit(client_combinations[i], cur_sum_list)
            #     self.group_top_k.extend(top_k_ind)
            #     print("top-k items in group {}: {}".format(client_combinations[i], top_k_ind))

            sum_time = time.time() - sum_start
            self.sum_completed = True

        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start

        # create response
        response_start = time.time()
        response = tenseal_shapley_data_pb2.top_k(
            client_rank=client_rank,
            ranking=self.group_top_k
        )
        response_time = time.time() - response_start

        # wait until creating all response
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == self.num_clients - 1:
            self.reset_sum()

        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        print(">>> server finish sum_shapley, cost {:.2f} s: deserialization {:.2f} s, "
              "wait for requests {:.2f} s, sum {:.2f} s, wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time,
                      wait_time, sum_time, sum_wait_time, response_time))

        return response


def launch_server(address, key_address, num_clients, k, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = ShapleyAggrKeyServer(address, key_address, num_clients, k, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_shapley_data_pb2_grpc.add_ShapleyServiceServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    server_address = args[0]
    key_server_address = args[1]
    num_clients = int(args[2])
    k = int(args[3])
    ctx_file = args[4]
    launch_server(server_address, key_server_address, num_clients, k, ctx_file)
