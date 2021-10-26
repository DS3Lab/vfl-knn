import time

import grpc
import numpy as np
import tenseal as ts

import sys
from transmission.tenseal import tenseal_data_pb2_grpc, tenseal_data_pb2


class Client:

    def __init__(self, server_address, client_rank, ctx_file):
        self.server_address = server_address
        self.client_rank = client_rank

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_data_pb2_grpc.SafeTransmissionStub(channel)

    def __sum_encrypted(self, plain_vector):
        print(">>> client sum encrypted start")

        # encrypt
        encrypt_start = time.time()
        enc_vector = ts.ckks_vector(self.ctx, plain_vector)
        encrypt_time = time.time() - encrypt_start

        print("size of msg: {} bytes".format(sys.getsizeof(enc_vector.serialize())))

        # create request
        request_start = time.time()
        request = tenseal_data_pb2.encrypted(
            client_rank=self.client_rank,
            msg=enc_vector.serialize()
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.sum_encrypted(request)
        comm_time = time.time() - comm_start

        # deserialize summed vector from response
        deserialize_start = time.time()
        assert self.client_rank == response.client_rank
        summed_encrypted_vector = ts.ckks_vector_from(self.ctx, response.msg)
        deserialize_time = time.time() - deserialize_start

        # decrypt vector
        decrypt_start = time.time()
        summed_plain_vector = summed_encrypted_vector.decrypt()
        decrypt_time = time.time() - decrypt_start

        print(">>> client sum encrypted end, cost {:.2f} s: encryption {:.2f} s, create request {:.2f} s, "
              "comm with server {:.2f} s, deserialize {:.2f} s, decryption {:.2f} s"
              .format(time.time() - encrypt_start, encrypt_time, request_time,
                      comm_time, deserialize_time, decrypt_time))

        return summed_plain_vector

    def transmit(self, plain_vector, operator="sum"):
        trans_start = time.time()
        received = self.__sum_encrypted(plain_vector) if operator == "sum" else None
        print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))
        return received


if __name__ == '__main__':
    server_address = "localhost:8991"
    ctx_file = "ts_ckks.config"
    plain_vector = np.arange(1000)
    client_rank = 0

    client = Client(server_address, client_rank, ctx_file)
    ret = client.transmit(plain_vector)

    print("plain vector, length {}, {}".format(len(plain_vector), plain_vector[:10]))
    print("sum vector, length {}, {}".format(len(ret), ret[:10]))
