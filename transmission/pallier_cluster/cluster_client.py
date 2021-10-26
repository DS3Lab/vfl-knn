from transmission.pallier_cluster import cluster_data_pb2, cluster_data_pb2_grpc

import grpc
from phe import paillier
import time
from transmission.encryption import decryptor
import numpy as np


# TODO: clean same in cluster DONE
class Client:

    def __init__(self, serverAddr, public_key, private_key):
        self.addr = serverAddr
        self.public_key = public_key
        self.private_key = private_key

        # Statistics arrays
        self.enc_timer = []
        self.extr_timer = []
        self.comm_timer = []
        self.recon_timer = []
        self.dec_timer = []

    # This method sends one message to server and receives it's response
    def __exchange_data(self, stub, trans_data_ids, trans_data_cids, trans_cids, trans_data):
        print("----client_exchangeData----")
        #print(trans_data[:5])

        print("num data = {} {}, num centroids = {} {}"
              .format(len(trans_data_ids), len(trans_data_cids), len(trans_cids), len(trans_data)))

        enc_start = time.time()
        enc_data = trans_data
        enc_time = time.time() - enc_start
        self.enc_timer.append(enc_time)
        print("Encryption took {} seconds".format(enc_time))

        extr_start = time.time()
        # int to str
        #cip_txts = list(map(lambda c: str(c.ciphertext()), enc_data))
        #exps = list(map(lambda c: str(c.exponent), enc_data))
        cip_txts = list(map(lambda c: c.ciphertext().to_bytes(572, 'little'), enc_data))
        exps = list(map(lambda c: c.exponent, enc_data))
        extr_time = time.time() - extr_start
        self.extr_timer.append(extr_time)
        print("Extraction took {} seconds".format(extr_time))

        comm_start = time.time()
        request = cluster_data_pb2.secret(
            dataIds=trans_data_ids,
            dataCIds=trans_data_cids,
            cIds=trans_cids,
            cipherTexts=cip_txts,
            exponents=exps
        )
        print("start exchange data, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = stub.exchangeData(request)
        comm_time = time.time() - comm_start
        self.comm_timer.append(comm_time)
        print("communication took {} seconds".format(comm_time))

        # str to int
        recon_start = time.time()
        cip_data = response.cipherText
        exp_data = response.exponent
        recon_data = [paillier.EncryptedNumber(self.public_key, int.from_bytes(cip, 'little'), exp)
                      for cip, exp in zip(cip_data, exp_data)]
        recon_time = time.time() - recon_start
        self.recon_timer.append(recon_time)
        print("Reconstruction took {} seconds".format(recon_time))

        dec_start = time.time()
        dec_data = decryptor(recon_data, self.private_key)
        dec_time = time.time() - dec_start
        self.dec_timer.append(dec_time)
        print("Decryption took {} seconds".format(dec_time))

        #print(dec_data[:5])
        print("----client_exchangeData_end----")

        return dec_data

    def transmit(self, trans_data_ids, trans_data_cids, trans_cids, trans_data, options):
        ret_data = []
        trans_start = time.time()

        with grpc.insecure_channel(self.addr, options=options) as channel:
            stub = cluster_data_pb2_grpc.safeTransmissionStub(channel)
            received = self.__exchange_data(stub,
                                            trans_data_ids, trans_data_cids,
                                            trans_cids, trans_data)
            ret_data.extend(received)
        trans_time = time.time() - trans_start
        print("transmission took {} seconds".format(trans_time))
        #print(ret_data[:5])
        return ret_data


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
