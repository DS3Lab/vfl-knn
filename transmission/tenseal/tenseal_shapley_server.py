import time
from concurrent import futures

import numpy as np
import grpc
import tenseal as ts

import sys
sys.path.append("../../")
from transmission.tenseal import tenseal_shapley_data_pb2, tenseal_shapley_data_pb2_grpc
from utils.shapley_utils import generate_all_combinations, sum_all_combinations


class ShapleyServer(tenseal_shapley_data_pb2_grpc.ShapleyServiceServicer):

    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients

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

        print("shapley server has been initialized")

    def reset_sum(self):
        self.sum_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

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

        # decrypt vector
        decrypt_start = time.time()
        dec_vector = enc_vector.decrypt()
        print("size of received vector: {}".format(len(dec_vector)))
        decrypt_time = time.time() - decrypt_start

        # add received data to cache
        self.sum_vectors.append(dec_vector)
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
            sum_time = time.time() - sum_start
            self.sum_completed = True

        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start

        client_combinations = generate_all_combinations(self.num_clients)
        top_k_combinations = []

        # find top-k for each summed list
        for i in range(len(self.sum_data)):
            cur_sum_list = self.sum_data[i]
            local_dist_ind = np.argsort(np.array(cur_sum_list))
            top_k_combinations.extend(local_dist_ind[:k])
            print("top-k items in group {}: {}"
                  .format(client_combinations[i], local_dist_ind[:k]))

        # create response
        response_start = time.time()
        response = tenseal_shapley_data_pb2.top_k(
            client_rank=client_rank,
            ranking=top_k_combinations
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

        print(">>> server finish sum_shapley, cost {:.2f} s: deserialization {:.2f} s, decryption {:.2f} s, "
              "wait for requests {:.2f} s, sum {:.2f} s, wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time, decrypt_time,
                      wait_time, sum_time, sum_wait_time, response_time))

        return response


def launch_server(address, num_clients, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = ShapleyServer(address, num_clients, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_shapley_data_pb2_grpc.add_ShapleyServiceServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    server_address = args[0]
    num_clients = int(args[1])
    ctx_file = args[2]
    launch_server(server_address, num_clients, ctx_file)
