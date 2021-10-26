import time
import math
import sys

import grpc
from phe import paillier
import numpy as np
from statistics import mean
from concurrent import futures

sys.path.append("../../")
from transmission.pallier import data_pb2, data_pb2_grpc
from transmission.encryption import encryptor


#TODO: same in cluster, even though not used
class Server(data_pb2_grpc.safeTransmissionServicer):
    # Server should be able to also transfer data on it or just work as server
    # --> can be server and client at same time or just server
    # If it is use as server only, server_data = []
    def __init__(self, addr, num_clients, public_key, server_data, split_size, exp_exch):
        self.addr = addr
        self.num_clients = num_clients
        self.public_key = public_key
        self.server_data = server_data
        self.split_size = split_size
        self.expect_exchange = exp_exch

        self.sleep_time = 0.1
        self.data = []
        self.sum_data = []
        self.expected_request = 0
        self.returned_response = 0
        self.n_exchanges = 0

        self.finish_last = True
        self.compute_sum = True
        self.sum_completed = False
        self.extract = True
        self.extract_completed = False
        self.exchange_time = 0
        self.extr_cip = []
        self.extr_exp = []
        # Statistics arrays
        self.coll_times = []
        self.recon_times = []
        self.store_times = []
        self.wait_times = []
        self.sum_times = []
        self.sum_wait_times = []
        self.extr_times = []
        self.resp_times = []

        if len(server_data) > 0:
            init_encr_start = time.time()
            # split server data
            t = [i for i in range(0, len(server_data), self.split_size)]
            self.server_data = [server_data[i: i + self.split_size] for i in range(0, len(server_data), self.split_size)]
            # instantiate array to add data from clients and server
            self.expected_request = self.num_clients + 1
            self.data.append(encryptor(self.server_data[0], self.public_key))
            init_encr_time = time.time() - init_encr_start
            #print(self.server_data[0][:5])
            print("initial Encryption took {} seconds".format(init_encr_time))
        else:
            # instantiate array to add data from clients only
            print("Server has no data")
            self.expected_request = self.num_clients

        print("server has been initialized")

    def __printStatistics(self):
        print("Exchange {}/{} was completed".format(self.n_exchanges, self.expect_exchange))
        print("*********************Exchange statistics*********************")
        if len(self.coll_times) > 0:
            print("Avg Collection Time {} with measurements {}".format(mean(self.coll_times), self.coll_times))
        if len(self.recon_times) > 0:
            print("Avg Reconstruction Time {} with measurements {}".format(mean(self.recon_times), self.recon_times))
        if len(self.store_times) > 0:
            print("Avg Appending Time {} with measurements {}".format(mean(self.store_times), self.store_times))
        if len(self.wait_times) > 0:
            print("Avg Waiting for appending Time {} with measurements {}".format(mean(self.wait_times), self.wait_times))
        if len(self.sum_times) > 0:
            print("Avg Summing Time {} with measurements {}".format(mean(self.sum_times), self.sum_times))
        if len(self.sum_wait_times) > 0:
            print("Avg Waiting for sum Time {} with measurements {}".format(mean(self.sum_wait_times), self.sum_wait_times))
        if len(self.extr_times) > 0:
            print("Avg Extraction Time {} with measurements {}".format(mean(self.extr_times), self.extr_times))
        if len(self.resp_times) > 0:
            print("Avg Response Time {} with measurements {}".format(mean(self.resp_times), self.resp_times))

    def __resetVars(self):
        self.data = []
        self.sum_data = []
        #self.expectedArrays = 0
        self.returned_response = 0
        self.compute_sum = True
        self.sum_completed = False
        self.extract = True
        self.extract_completed = False
        self.exchange_time = 0
        self.extr_cip = []
        self.extr_exp = []
        # Measurements per exchange
        self.coll_times = []
        self.recon_times = []
        self.extr_times = []
        self.wait_times = []
        self.sum_wait_times = []
        self.resp_times = []

    def is_reset(self):
        return len(self.resp_times) == 0

    def extractCip(self, data):
        return list(map(lambda c: str(c.ciphertext()), data))

    def extractExp(self, data):
        return list(map(lambda c: str(c.exponent), data))

    def exchangeData(self, request, context):

        print(">>> server starts exchange data, time = {} ----"
              .format(time.asctime(time.localtime(time.time()))))

        if len(self.server_data) > 0:
            print(self.server_data[self.n_exchanges][:5])

        exchange_start = time.time()

        # collect data from clients: ciphertexts and exponents
        coll_start = time.time()
        cip_txts = request.cipherText
        exps = request.exponent
        coll_time = time.time() - coll_start
        self.coll_times.append(coll_time)

        # use collected data to reconstruct encrypted objects
        recon_start = time.time()
        # str to int
        recon_data = [paillier.EncryptedNumber(self.public_key, int.from_bytes(cip, 'little'), exp)
                      for cip, exp in zip(cip_txts, exps)]
        recon_time = time.time() - recon_start
        self.recon_times.append(recon_time)

        # set counters to verify that data from all clients has been received correctly
        # store reconstructed data from client
        store_start = time.time()
        self.data.append(recon_data)
        store_time = time.time() - store_start
        self.store_times.append(store_time)

        request_count = len(self.data)

        # wait for receiving of all clients' requests
        wait_start = time.time()
        while request_count != self.expected_request:
            time.sleep(self.sleep_time)
            request_count = len(self.data)
        wait_time = time.time() - wait_start
        self.wait_times.append(wait_time)

        # once have reconstructed data from all clients, apply sum
        # use flags, so sum is computed only once
        sum_time = 0
        if self.compute_sum:
            self.compute_sum = False  # so others clients don't enter here
            sum_start = time.time()
            self.sum_data = np.sum(self.data, axis=0)
            sum_time = time.time() - sum_start
            self.sum_times.append(sum_time)
            self.sum_completed = True

        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start
        self.sum_wait_times.append(sum_wait_time)

        # extract ciphertexts and exponents form summed data
        # use flag so that extraction is performed only once
        extr_time = 0
        if self.extract:
            self.extract = False
            extr_start = time.time()
            print(len(self.sum_data))

            # with Pool() as pool:
            #     self.extrCip = pool.map(self.extractCip, [self.sumData])[0]
            #     self.extrExp = pool.map(self.extractExp, [self.sumData])[0]

            self.extr_cip = [i.ciphertext().to_bytes(572, 'little') for i in self.sum_data]
            self.extr_exp = [i.exponent for i in self.sum_data]

            self.extract_completed = True
            extr_time = time.time() - extr_start
            self.extr_times.append(extr_time)

        while not self.extract_completed:
            time.sleep(self.sleep_time)

        # send back the extracted data to the clients
        # use counter to count the returned responses
        response_start = time.time()
        response = data_pb2.globalDist(
            cipherText=self.extr_cip,
            exponent=self.extr_exp
        )
        response_time = time.time() - response_start
        self.resp_times.append(response_time)

        self.__printStatistics()
        self.returned_response = self.returned_response + 1

        while self.returned_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # Once we received the expected number of arrays and returned a response to every client
        # We reset all counters and variable to proceed with exchange of next split
        while not self.is_reset():
            self.__resetVars()

        self.n_exchanges = self.n_exchanges + 1
        while self.n_exchanges % self.num_clients != 0:
            time.sleep(self.sleep_time)

        # if self.n_exchanges < self.expect_exchange and len(self.server_data) > 0:
        #     init_encr_start = time.time()
        #     self.data.append(encryptor(self.server_data[self.n_exchanges], self.public_key))
        #     init_encr_time = time.time() - init_encr_start
        #     print("initial encryption took {} seconds".format(init_encr_time))
        #     #print(self.serverData[self.exchanges][:5])

        exchange_time = time.time() - exchange_start

        print("collection of client request took {} seconds".format(coll_time))
        print("reconstruction of encrypted data took {} seconds".format(recon_time))
        print("store array {} took {} seconds".format(len(self.data), store_time))
        print("wait {} seconds for {} arrays".format(wait_time, len(self.data)))
        print("sum of {} arrays took {} seconds".format(len(self.data), sum_time))
        print("wait {} seconds for sum to be completed".format(sum_wait_time))
        print("extraction took {} seconds".format(extr_time))
        print("transmission of response {} took {} seconds".format(self.returned_response, response_time))

        print(">>> server ends exchange data, cost {:.2f} s, collect request cost {:.2f} s, "
              "reconstruct encrypted data {:.2f} s, store encrypted data {:.2f} s, wait {} clients {:.2f} s, "
              "sum {} arrays {:.2f} seconds, wait sum {:.2f} s, extraction {} s, transmit response {} s"
              .format(exchange_time, coll_time, recon_time, store_time, len(self.data), wait_time,
                      len(self.data), sum_time, sum_wait_time, extr_time, response_time))

        return response


def broadcast(addr, numClients, public_key, serverData, splitSize, expExch):
    max_msg_size = 80000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = Server(addr, numClients, public_key, serverData, splitSize, expExch)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    data_pb2_grpc.add_safeTransmissionServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    size = int(args[0])
    splitSize = int(args[1])
    expExch = math.ceil(size/splitSize)
    addr = args[2]
    numClients = int(args[3])
    # In order to keep this key static every time we run this scrip, we set cst values
    n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
    public_key = paillier.PaillierPublicKey(n=n)
    serverData = [] #np.random.rand(1, size)[0]
    broadcast(addr, numClients, public_key, serverData, splitSize, expExch)

