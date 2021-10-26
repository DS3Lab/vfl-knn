from transmission.pallier import data_pb2, data_pb2_grpc
import  grpc
from phe import paillier
import time
from transmission.encryption import decryptor
import numpy as np


class Client:

    def __init__(self, serverAddr, public_key, private_key):
        self.addr = serverAddr
        self.public_key = public_key
        self.private_key = private_key
        # Statistics arrays
        self.encr_times = []
        self.extr_times = []
        self.request_times = []
        self.comm_times = []
        self.response_times = []
        self.recon_times = []
        self.decr_times = []

    def __exchange_data(self, stub, split):
        # This method will take care of senidn one message to server
        # and receiving it's response
        print(">>> client exchangeData start")
        #print(split[:5])
        exchange_start = time.time()

        encr_start = time.time()
        encr_data = split
        encr_time = time.time() - encr_start
        self.encr_times.append(encr_time)

        extr_start = time.time()
        # int to str
        # cipTxts = list(map(lambda c: str(c.ciphertext()), encrData))
        # exps = list(map(lambda c: str(c.exponent), encrData))
        cipT_txts = list(map(lambda c: c.ciphertext().to_bytes(572, 'little'), encr_data))
        exps = list(map(lambda c: c.exponent, encr_data))
        extr_time = time.time() - extr_start
        self.extr_times.append(extr_time)

        request_start = time.time()
        request = data_pb2.secret(
            cipherText=cipT_txts,
            exponent=exps,
        )
        request_time = time.time() - request_start
        self.request_times.append(request_time)

        comm_start = time.time()
        print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = stub.exchangeData(request)
        comm_time = time.time() - comm_start
        self.comm_times.append(comm_time)

        response_start = time.time()
        cip_response = response.cipherText
        exp_response = response.exponent
        response_time = time.time() - response_start
        self.response_times.append(response_time)

        recon_start = time.time()
        # str to int
        recon_data = [paillier.EncryptedNumber(self.public_key, int.from_bytes(cip, 'little'), exp)
                     for cip, exp in zip(cip_response, exp_response)]
        recon_time = time.time() - recon_start
        self.recon_times.append(recon_time)

        decr_start = time.time()
        decr_data = decryptor(recon_data, self.private_key)
        decr_time = time.time() - decr_start
        self.decr_times.append(decr_time)

        exchange_time = time.time() - exchange_start
        #print(decrData[:5])
        print(">>> client exchangeData end, cost {:.2f} s, encryption {:.2f} s, extraction {:.2f} s, construct request {:.2f} s, "
              "comm with server {:.2f} s, parse response {:.2f} s, reconstruct response {:.2f} s, decryption {:.2f} s"
              .format(exchange_time, encr_time, extr_time, request_time, comm_time, response_time, recon_time, decr_time))

        return decr_data

    def transmit(self, data, split_size, options):
        # This method will take care of the whole transmission and receptions
        # It will pass the data split by split to the __exchange_data method
        global_data = []
        # With splitSize this method splits data to transmit
        split_data = [data[i : i + split_size] for i in range(0, len(data), split_size)]
        count = 0
        transmS = time.time()

        with grpc.insecure_channel(self.addr, options=options) as channel:
            for split in split_data:
                exchS = time.time()
                count = count + 1
                stub = data_pb2_grpc.safeTransmissionStub(channel)
                received = self.__exchange_data(stub, split)
                global_data.extend(received)
                exchE = time.time() - exchS
                #print("exchange {} took {} seconds".format(count, exchE))
        transmE = time.time() - transmS
        print("transmission of {} messages took {} seconds".format(count, transmE))
        #print(globData[:5])
        return global_data


if __name__ == '__main__':
    n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
    p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
    q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
    size = 6000
    splitSize=2000
    max_msg_size = 80000000
    public_key = paillier.PaillierPublicKey(n=n)
    private_key = paillier.PaillierPrivateKey(public_key, p, q)
    serverAddr = 'localhost:8890'
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    clientData = np.random.rand(1, size)[0]
    client = Client(serverAddr, public_key, private_key)
    client.transmit(clientData, splitSize, options)
