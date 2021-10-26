import time

import grpc
import numpy as np
import tenseal as ts

import sys
from . import tenseal_key_server_pb2_grpc, tenseal_key_server_pb2


class KeyServerClient:

    def __init__(self, server_address, k, ctx_file):
        self.server_address = server_address
        self.k = k
        self.ctx_file = ctx_file

        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_key_server_pb2_grpc.KeyServerServiceStub(channel)

    def __top_k(self, group_key, enc_vector):
        print(">>> client sum encrypted start")

        # create request
        request_start = time.time()
        enc_vector_bytes = enc_vector.serialize()
        print("size of msg: {} bytes".format(sys.getsizeof(enc_vector_bytes)))
        request = tenseal_key_server_pb2.key_server_msg(
            group_key=group_key,
            k=self.k,
            msg=enc_vector_bytes
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.find_top_k(request)
        comm_time = time.time() - comm_start

        # get top-k
        assert group_key == response.group_key
        top_k_ind = list(response.ranking)
        assert len(top_k_ind) == self.k

        print(">>> key server find top-k end, cost {:.2f} s: create request {:.2f} s, comm with key server {:.2f} s,"
              .format(time.time() - request_start, request_time, comm_time))

        return top_k_ind

    def transmit(self, group_key, enc_vector):
        trans_start = time.time()
        response = self.__top_k(group_key, enc_vector)
        print(">>> client transmission cost {:.2f} s, return top-k ranking {}"
              .format(time.time() - trans_start, response))
        return response
