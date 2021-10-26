import time

import grpc
import numpy as np
import tenseal as ts

import sys
from . import tenseal_shapley_data_pb2_grpc, tenseal_shapley_data_pb2


class ShapleyClient:

    def __init__(self, server_address, args):
        self.server_address = server_address
        self.client_rank = args.rank
        self.num_clients = args.world_size
        self.k = args.k
        self.ctx_file = args.config

        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_shapley_data_pb2_grpc.ShapleyServiceStub(channel)

    def __sum_shapley(self, plain_vector):
        print(">>> client sum encrypted start")

        # encrypt
        encrypt_start = time.time()
        enc_vector = ts.ckks_vector(self.ctx, plain_vector)
        encrypt_time = time.time() - encrypt_start

        # create request
        request_start = time.time()
        enc_vector_bytes = enc_vector.serialize()
        print("size of msg: {} bytes".format(sys.getsizeof(enc_vector_bytes)))
        request = tenseal_shapley_data_pb2.client_shapley_msg(
            client_rank=self.client_rank,
            k=self.k,
            msg=enc_vector_bytes
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.sum_shapley(request)
        comm_time = time.time() - comm_start

        # deserialize summed vector from response
        deserialize_start = time.time()
        assert self.client_rank == response.client_rank
        top_k_ranking_flat = list(response.ranking)
        assert len(top_k_ranking_flat) == (2**self.num_clients - 1) * self.k
        deserialize_time = time.time() - deserialize_start

        print(">>> client sum shapley end, cost {:.2f} s: encryption {:.2f} s, create request {:.2f} s, "
              "comm with server {:.2f} s, deserialize {:.2f} s"
              .format(time.time() - encrypt_start, encrypt_time, request_time,
                      comm_time, deserialize_time))

        return top_k_ranking_flat

    def transmit(self, plain_vector, operator="sum"):
        trans_start = time.time()
        response = self.__sum_shapley(plain_vector) if operator == "sum" else None
        print(">>> client transmission cost {:.2f} s, return top-k ranking {}"
              .format(time.time() - trans_start, response[:10]))
        return response
