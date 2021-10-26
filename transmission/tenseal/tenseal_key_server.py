import time
from concurrent import futures

import numpy as np
import grpc
import tenseal as ts

import sys
sys.path.append("../../")
from transmission.tenseal import tenseal_key_server_pb2, tenseal_key_server_pb2_grpc


class KeyServer(tenseal_key_server_pb2_grpc.KeyServerServiceServicer):

    def __init__(self, address, ctx_file):
        self.address = address

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        print("key server has been initialized")

    def find_top_k(self, request, context):

        server_start = time.time()

        group_key = request.group_key
        k = request.k

        print(">>> key server receive encrypted data from aggregation server, "
              "group ={}, k = {}, time = {} ----"
              .format(group_key, k, time.asctime(time.localtime(time.time()))))

        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start

        # decrypt vector
        decrypt_start = time.time()
        dec_vector = enc_vector.decrypt()
        print("size of received vector: {}".format(len(dec_vector)))
        decrypt_time = time.time() - decrypt_start

        local_dist_ind = np.argsort(np.array(dec_vector))
        top_k_ind = local_dist_ind[:k]
        print("top-k items in group {}: {}".format(group_key, top_k_ind))

        # create response
        response_start = time.time()
        response = tenseal_key_server_pb2.top_k_key_server(
            group_key=group_key,
            ranking=top_k_ind
        )
        response_time = time.time() - response_start

        print(">>> key server finishes find top k for group {}, top {} = {}, cost {:.2f} s, "
              "deserialize cost {:.2f} s, decrypt cost {:.2f} s, "
              .format(group_key, k, top_k_ind, time.time() - server_start, deser_time, decrypt_time))

        return response


def launch_server(address, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = KeyServer(address, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_key_server_pb2_grpc.add_KeyServerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    server_address = args[0]
    ctx_file = args[1]
    launch_server(server_address, ctx_file)
