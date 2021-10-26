import time
import sys

import grpc
from phe import paillier
from statistics import mean
from concurrent import futures

import threading

sys.path.append("..")
from transmission.pallier_cluster import cluster_data_pb2, cluster_data_pb2_grpc


#TODO: same in cluster, even though not used
class Server(cluster_data_pb2_grpc.safeTransmissionServicer):

    # Server should be able to also transfer data on it or just work as server
    # --> can be server and client at same time or just server
    # If it is use as server only, serverData = []
    def __init__(self, addr, num_clients, num_test, public_key):
        self.addr = addr
        self.num_clients = num_clients
        self.public_key = public_key
        self.num_test = num_test
        self.sleepTime = 1

        # do not need to reset
        self.n_exchanges = 0
        self.lock = threading.Lock()

        # need to reset
        self.num_data = 0
        self.num_cluster = 0

        self.centroids = [] # num_client * num_cluster
        self.data2centroids = [] # n_data x n_client
        self.sum_data = []
        self.ret_cipher = []
        self.ret_exp = []

        self.n_received_request = 0
        self.n_returned_response = 0
        self.num_summed = 0

        # statistics arrays
        self.exchange_time = 0
        self.recon_timer = []
        self.wait_timer = []
        self.sum_timer = []
        self.sum_wait_timer = []
        self.resp_timer = []

        print("server has been initialized")

    def __printStatistics(self):
        print("Exchange {}/{} was completed".format(self.n_exchanges, self.num_test))
        print("This exchange with {} clients took {} seconds".format(self.num_clients, self.exchange_time))
        print("*********************Exchange statistics*********************")
        print("Avg Reconstruction Time {} with measurements {}".format(mean(self.recon_timer), self.recon_timer))
        print("Avg Waiting for appending Time {} with measurements {}".format(mean(self.wait_timer), self.wait_timer))
        print("Avg Summing Time {} with measurements {}".format(mean(self.sum_timer), self.sum_timer))
        print("Avg Waiting for sum Time {} with measurements {}".format(mean(self.sum_wait_timer), self.sum_wait_timer))
        print("Avg Response Time {} with measurements {}".format(mean(self.resp_timer), self.resp_timer))

    def __resetVars(self):
        self.num_data = 0
        self.num_cluster = 0

        self.centroids = []  # num_client * num_cluster
        self.data2centroids = []  # n_data x n_client
        self.sum_data = []
        self.ret_cipher = []
        self.ret_exp = []

        self.n_received_request = 0
        self.n_returned_response = 0
        self.num_summed = 0

        # statistics arrays
        self.exchange_time = 0
        self.recon_timer = []
        self.wait_timer = []
        self.sum_timer = []
        self.sum_wait_timer = []
        self.resp_timer = []

    def extractCip(self, data):
        return list(map(lambda c: str(c.ciphertext()), data))

    def extractExp(self, data):
        return list(map(lambda c: str(c.exponent), data))

    def exchangeData(self, request, context):

        print("----Starting exchange, time = {} ----"
              .format(time.asctime(time.localtime(time.time()))))

        exchange_start = time.time()

        # parse message
        data_ids = request.dataIds
        data_cids = request.dataCIds
        centroid_ids = request.cIds
        cip_txts = request.cipherTexts
        exps = request.exponents
        assert len(data_ids) == len(data_cids)
        assert len(centroid_ids) == len(cip_txts) and len(cip_txts) == len(exps)

        print("{} data ids: {}".format(len(data_ids), data_ids[:10]))
        print("{} data cids: {}".format(len(data_cids), data_cids[:10]))
        print("{} centroid ids: {}".format(len(centroid_ids), centroid_ids[:10]))
        print("{} cip_txts, {} exps".format(len(cip_txts), len(exps)))

        # init cluster number
        if self.num_cluster == 0:
            self.num_cluster = len(centroid_ids)
        # init data number
        if self.num_data == 0:
            self.num_data = len(data_ids)
        # client rank, the master is not included
        client_rank = int(centroid_ids[0].split("_")[0]) - 1
        print("client rank = {}".format(client_rank))

        # use received data to reconstruct encrypted objects
        recon_start = time.time()
        # str to int
        recon_data = [paillier.EncryptedNumber(self.public_key, int.from_bytes(cip, 'little'), exp)
                      for cip, exp in zip(cip_txts, exps)]
        recon_time = time.time() - recon_start
        self.recon_timer.append(recon_time)
        print("Reconstruction of encrypted data took {} seconds".format(recon_time))

        # init centroids
        if len(self.centroids) == 0:
            self.lock.acquire()
            self.centroids = [0] * (self.num_clients * self.num_cluster)
            self.lock.release()
        assert len(self.centroids) == self.num_clients * self.num_cluster
        print("centroids length = {}".format(len(self.centroids)))

        # store centroids
        start_centroid_id = client_rank * self.num_cluster
        for i in range(len(centroid_ids)):
            self.centroids[start_centroid_id + i] = recon_data[i]
        print("finish store centroids")

        # init data2centroids
        if len(self.data2centroids) != self.num_data:
            self.lock.acquire()
            self.data2centroids = [[0] * self.num_clients] * self.num_data
            self.lock.release()
        print("finish init data2centroids")

        # update data's centroid ids
        for i in range(self.num_data):
            data_cid = data_cids[i]
            local_cluster_ind = int(data_cid.split("_")[1])
            # rank 0 is master, not included
            self.data2centroids[i][client_rank] = start_centroid_id + local_cluster_ind
        print("finish update data2centroids")

        self.n_received_request += 1

        # wait until receiving requests from all clients
        wait_start = time.time()
        while self.n_received_request != self.num_clients:
            time.sleep(self.sleepTime)
        wait_time = time.time() - wait_start
        self.wait_timer.append(wait_time)
        print("num clients = {}, num candidates = {}, num clusters = {}, num centroids = {}"
              .format(self.num_clients, self.num_data, self.num_cluster, len(self.centroids)))
        print("wait {} seconds until receiving requests from {} clients ".format(wait_time, self.num_clients))

        # once receiving data from all clients, apply sum and extract cipher texts and exponents form summed data
        # init sum response
        sum_start = time.time()
        self.lock.acquire()
        if len(self.sum_data) == 0:
            self.sum_data = [0] * self.num_data
            self.ret_cipher = [int(0).to_bytes(572, 'little')] * self.num_data
            self.ret_exp = [0] * self.num_data
        print("finish init sum response")
        self.lock.release()

        num_sum = self.num_data // self.num_clients
        if self.num_data % self.num_clients != 0:
            num_sum += 1
        print("num sum = {}".format(num_sum))

        # responsible data range
        start_data_id = client_rank * num_sum
        end_data_id = min((client_rank + 1) * num_sum, self.num_data)

        # sum encrypted data
        for i in range(start_data_id, end_data_id):
            sum_item = self.centroids[self.data2centroids[i][0]]
            for j in range(1, self.num_clients):
                sum_item += self.centroids[self.data2centroids[i][j]]
            self.sum_data[i] = sum_item
            # self.ret_cipher[i] = str(sum_item.ciphertext())
            # self.ret_exp[i] = str(sum_item.exponent)
        sum_time = time.time() - sum_start
        self.sum_timer.append(sum_time)
        print("sum data from {} to {} took {} seconds, {} have summed"
              .format(start_data_id, end_data_id, sum_time, self.num_summed))

        # extract encrypted data
        extract_start = time.time()
        for i in range(start_data_id, end_data_id):
            self.ret_cipher[i] = self.sum_data[i].ciphertext().to_bytes(572, 'little')
            self.ret_exp[i] = self.sum_data[i].exponent
        extract_time = time.time() - extract_start
        print("extract encrypted data cost {:.2f} seconds".format(extract_time))

        self.num_summed += end_data_id - start_data_id

        sum_wait_start = time.time()
        while self.num_summed != self.num_data:
            time.sleep(self.sleepTime)
        sum_wait_time = time.time() - sum_wait_start
        self.sum_wait_timer.append(sum_wait_time)
        print("wait {} seconds for sum and extract to be completed".format(sum_wait_time))

        # send back extracted data to the clients
        # We use counter to count the returned responses
        response_start = time.time()
        response = cluster_data_pb2.globalDist(
            dataIds=data_ids,
            cipherText=self.ret_cipher,
            exponent=self.ret_exp
        )
        self.n_returned_response += 1
        response_time = time.time() - response_start
        self.resp_timer.append(response_time)
        print("transmission of response {} took {} seconds".format(self.n_returned_response, response_time))

        if self.n_returned_response == self.num_clients and self.n_received_request == self.num_clients:
            # once received request from and returned response to every client
            # reset all counters and variable
            self.exchange_time = time.time() - exchange_start
            self.n_exchanges = self.n_exchanges + 1
            self.__printStatistics()
            self.__resetVars()

        return response


def broadcast(addr, num_clients, num_test, public_key):
    max_msg_size = 80000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = Server(addr, num_clients, num_test, public_key)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    cluster_data_pb2_grpc.add_safeTransmissionServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    addr = args[0]
    num_clients = int(args[1])
    num_test = int(args[2])
    # In order to keep this key static every time we run this scrip, we set cst values
    n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
    public_key = paillier.PaillierPublicKey(n=n)
    broadcast(addr, num_clients, num_test, public_key)
