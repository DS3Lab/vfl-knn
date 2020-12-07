import time
import grpc
import data_pb2, data_pb2_grpc
import numpy as np
from encryption import encryptor
from phe import paillier
from statistics import mean
from concurrent import futures
import math, sys
from multiprocessing import Pool
#TODO: same in cluster, even though not used
class Server(data_pb2_grpc.safeTransmissionServicer):
    # Server should be able to also transfer data on it or just work as server
    # --> can be server and client at same time or just server
    # If it is use as server only, serverData = []
    def __init__(self, addr, numClients, public_key, serverData, splitSize, expExch):
        self.addr = addr
        self.numClients = numClients
        self.public_key = public_key
        self.serverData = serverData
        self.splitSize = splitSize
        self.expExch = expExch
        self.sleepTime = 1
        self.data = []
        self.sumData = []
        self.expectedArrays = 0
        self.returnedResponses = 0
        self.exchanges = 0
        self.computeSum = True
        self.sumCompleted = False
        self.extract = True
        self.extractCompleted = False
        self.exchE = 0
        self.extrCip = []
        self.extrExp = []
        # Statistics arrays
        self.collTimes = []
        self.reconTimes = []
        self.appTimes = []
        self.waitTimes = []
        self.sumTimes = []
        self.sumWaitTimes = []
        self.extrTimes = []
        self.respTimes = []

        if len(serverData) > 0:
            initEncrS = time.time()
            # split server data
            t = [i for i in range(0, len(serverData), self.splitSize)]
            self.serverData = [serverData[i: i + self.splitSize] for i in range(0, len(serverData), self.splitSize)]
            # instantiate array to add data from clients and server
            self.expectedArrays = self.numClients + 1
            self.data.append(encryptor(self.serverData[0], self.public_key))
            initEncrE = time.time() - initEncrS
            #print(self.serverData[0][:5])
            print("Initial Encryption took {} seconds".format(initEncrE))
        else:
            # instantiate array to add data from clients only
            print("Server has no data")
            self.expectedArrays = self.numClients

        print("server has been initialized")

    def __printStatistics(self):
        print("Exchange {}/{} was completed".format(self.exchanges, self.expExch))
        print("This exchange with {} clients took {} seconds".format(self.numClients, self.exchE))
        print("*********************Exchange statistics*********************")
        print("Avg Collection Time {} with measurements {}".format(mean(self.collTimes), self.collTimes))
        print("Avg Reconstruction Time {} with measurements {}".format(mean(self.reconTimes), self.reconTimes))
        print("Avg Appending Time {} with measurements {}".format(mean(self.appTimes), self.appTimes))
        print("Avg Waiting for appending Time {} with measurements {}".format(mean(self.waitTimes), self.waitTimes))
        print("Avg Summing Time {} with measurements {}".format(mean(self.sumTimes), self.sumTimes))
        print("Avg Waiting for sum Time {} with measurements {}".format(mean(self.sumWaitTimes), self.sumWaitTimes))
        print("Avg Extraction Time {} with measurements {}".format(mean(self.extrTimes), self.extrTimes))
        print("Avg Response Time {} with measurements {}".format(mean(self.respTimes), self.respTimes))

    def __resetVars(self):
        self.data = []
        self.sumData = []
        #self.expectedArrays = 0
        self.returnedResponses = 0
        self.computeSum = True
        self.sumCompleted = False
        self.extract = True
        self.extractCompleted = False
        self.exchE = 0
        self.extrCip = []
        self.extrExp = []
        # Measurements per exchange
        self.collTimes = []
        self.reconTimes = []
        self.extrTimes = []
        self.respTimes = []
        self.waitTimes = []
        self.sumWaitTimes = []

    def extractCip(self, data):
        return list(map(lambda c: str(c.ciphertext()), data))

    def extractExp(self, data):
        return list(map(lambda c: str(c.exponent), data))

    def exchangeData(self, request, context):

        print("----Starting exchange----")
        if len(self.serverData) > 0:
            print(self.serverData[self.exchanges][:5])

        exchS = time.time()

        # We collect data from clients: ciphertexts and exponents
        collS = time.time()
        cipTxts = request.cipherText
        exps = request.exponent
        collE = time.time() - collS
        self.collTimes.append(collE)
        print("Collection of client data took {} seconds".format(collE))

        # We use collected data to reconstruct encrypted objects
        reconS = time.time()
        # str to int
        reconData = [paillier.EncryptedNumber(self.public_key, int(cip), int(exp)) for cip, exp in zip(cipTxts, exps)]
        reconE = time.time() - reconS
        self.reconTimes.append(reconE)
        print("Reconstruction of encrypted data took {} seconds".format(reconE))

        # We append reconstructed data from all clients into one array

        # We set counters to verify that data from all clients has been received correctly

        # We append reconstructed data from client
        appS = time.time()
        self.data.append(reconData)
        appE = time.time() - appS
        self.appTimes.append(appE)
        print("Appending of array {} took {} seconds".format(len(self.data), appE))

        recCount = len(self.data)

        # Loops to wait for reception of all clients data
        waitS = time.time()
        while recCount != self.expectedArrays:
            time.sleep(self.sleepTime)
            recCount = len(self.data)
        waitE = time.time() - waitS
        self.waitTimes.append(waitE)
        print("We waited {} seconds for array {}".format(waitE, len(self.data)))

        # Once we have reconstructed data from all clients
        # We apply sum

        # We use flags, so sum is computed only once
        if self.computeSum:
            self.computeSum = False  # so others clients don't enter here
            sumS = time.time()
            self.sumData = np.sum(self.data, axis=0)
            sumE = time.time() - sumS
            self.sumTimes.append(sumE)
            print("Sum of {} arrays took {} seconds".format(len(self.data), sumE))
            self.sumCompleted = True
        else:
            print("Sum is being computed")

        sumWaitS = time.time()
        while not self.sumCompleted:
            time.sleep(self.sleepTime)
        sumWaitE = time.time() - sumWaitS
        self.sumWaitTimes.append(sumWaitE)
        print("We waited {} seconds for sum to be completed".format(sumWaitE))

        # We now extract ciphertexts and exponents form summed data

        # We use flags, so extraction is performed only once
        if self.extract:
            self.extract = False
            extrS = time.time()
            print(len(self.sumData))

            with Pool() as pool:
                self.extrCip = pool.map(self.extractCip, [self.sumData])[0]
                self.extrExp = pool.map(self.extractExp, [self.sumData])[0]


            self.extractCompleted = True
            extrE = time.time() - extrS
            self.extrTimes.append(extrE)
            print("Extraction took {} seconds".format(extrE))
        else:
            print("Extraction is being completed")

        while not self.extractCompleted:
            time.sleep(self.sleepTime)

        # We now send back the extracted data to the clients

        # We use counter to count the returned responses
        respS = time.time()
        response = data_pb2.globalDist(
            cipherText=self.extrCip,
            exponent=self.extrExp
        )
        self.returnedResponses = self.returnedResponses + 1
        respE = time.time() - respS
        self.respTimes.append(respE)
        print("Transmission of response {} took {} seconds".format(self.returnedResponses, respE))

        if self.returnedResponses == self.numClients and recCount == self.expectedArrays:
            # Once we received the expected number of arrays and returned a response to every client
            # We reset all counters and variable to proceed with exchange of next split
            self.exchE = time.time() - exchS
            self.exchanges = self.exchanges + 1
            self.__printStatistics()
            self.__resetVars()
            if self.exchanges < self.expExch and len(self.serverData) > 0:
                initEncrS = time.time()
                self.data.append(encryptor(self.serverData[self.exchanges], self.public_key))
                initEncrE = time.time() - initEncrS
                print("Initial Encryption took {} seconds".format(initEncrE))
                #print(self.serverData[self.exchanges][:5])
        return response


def broadcast(addr, numClients, public_key, serverData, splitSize, expExch):
    max_msg_size = 80000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = Server(addr,numClients,public_key, serverData, splitSize, expExch)
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
    addr = 'bach08.ethz.ch:'+args[2]
    numClients = int(args[3])
    # In order to keep this key static every time we run this scrip, we set cst values
    n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
    public_key = paillier.PaillierPublicKey(n=n)
    serverData = []#np.random.rand(1, size)[0]
    broadcast(addr, numClients, public_key, serverData, splitSize, expExch)

