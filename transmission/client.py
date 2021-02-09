from transmission import data_pb2, data_pb2_grpc
import  grpc
from phe import paillier
import time
from transmission.encryption import decryptor
import numpy as np

#TODO: clean same in cluster DONE
class Client:

    def __init__(self, serverAddr, public_key, private_key):
        self.addr = serverAddr
        self.public_key = public_key
        self.private_key = private_key
        # Statistics arrays
        self.encrTimes = []
        self.extrTimes = []
        self.sendTimes = []
        self.recvTimes = []
        self.storTimes = []
        self.reconTimes = []
        self.decrTimes = []

    def __exchange_data(self, stub, split):
        # This method will take care of senidn one message to server
        # and receiving it's response
        print("----client_exchangeData----")
        #print(split[:5])

        encrS = time.time()
        encrData = split
        encrE = time.time() - encrS
        self.encrTimes.append(encrE)
        print("Encryption took {} seconds".format(encrE))

        extrS = time.time()
        # int to str
        cipTxts = list(map(lambda c: str(c.ciphertext()), encrData))
        exps = list(map(lambda c: str(c.exponent), encrData))
        extrE = time.time() - extrS
        self.extrTimes.append(extrE)
        print("Extraction took {} seconds".format(extrE))

        sendS = time.time()
        request = data_pb2.secret(
            cipherText=cipTxts,
            exponent=exps,
        )

        sendE = time.time() - sendS
        self.sendTimes.append(encrE)
        print("Sending took {} seconds".format(sendE))

        recvS = time.time()
        response = stub.exchangeData(request)
        recvE = time.time() - recvS
        self.recvTimes.append(recvE)
        print("Reception took {} seconds".format(recvE))

        storS = time.time()
        cipData=response.cipherText
        expData=response.exponent
        storE = time.time()- storS
        self.storTimes.append(storE)
        print("Storage took {} seconds".format(storE))

        reconS = time.time()
        # str to int
        reconData = [paillier.EncryptedNumber(self.public_key,int(cip),int(exp)) for cip, exp in zip(cipData, expData)]
        reconE = time.time() - reconS
        self.reconTimes.append(reconE)
        print("Reconstruction took {} seconds".format(reconE))

        decrS = time.time()
        decrData = decryptor(reconData,self.private_key)
        decrE = time.time() - decrS
        self.decrTimes.append(decrE)
        print("Decryption took {} seconds".format(decrE))

        print(decrData[:5])
        print("----client_exchangeData_end----")

        return decrData




    def transmit(self, data, splitSize, options):
        # This method will take care of the whole transmission
        # and receptions
        # It will pass the data split by split to the __exchange_data method
        globData=[]
        #With splitSize this method splits data to transmit
        splitData = [data[i : i + splitSize] for i in range(0, len(data), splitSize)]
        count = 0
        transmS = time.time()

        with grpc.insecure_channel(self.addr, options=options) as channel:
            for split in splitData:
                exchS = time.time()
                count = count + 1
                stub = data_pb2_grpc.safeTransmissionStub(channel)
                received = self.__exchange_data(stub,split)
                globData.extend(received)
                exchE = time.time() - exchS
                print("Exchange {} took {} seconds".format(count, exchE))
        transmE = time.time() - transmS
        print("Transmission of {} messages took {} seconds".format(count, transmE))
        #print(globData[:5])
        return globData


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
