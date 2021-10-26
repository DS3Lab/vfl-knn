import sys
import time
import math

from phe import paillier

if __name__ == '__main__':
    n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
    # p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
    # q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
    # public_key = paillier.PaillierPublicKey(n=n)
    # private_key = paillier.PaillierPrivateKey(public_key, p, q)

    print(len(str(n)))
    print(math.log2(n))

    public_key, private_key = paillier.generate_paillier_keypair()

    enc = public_key.encrypt(0.01)
    cip = enc.ciphertext()
    print(math.log2(cip))
    exp = enc.exponent

    print(cip)
    print(exp)
    print("type of cip: {}".format(type(cip)))
    print("size of cip: {}".format(sys.getsizeof(cip)))
    print("type of exp: {}".format(type(exp)))
    print("size of exp: {}".format(sys.getsizeof(exp)))

    cip_str = str(enc.ciphertext())
    print("cip str len = {}".format(sys.getsizeof(cip_str)))

    cip_bytes = enc.ciphertext().to_bytes(572, 'little')
    print("cip bytes len = {}".format(sys.getsizeof(cip_bytes)))

    cip_bytes_int = int.from_bytes(cip_bytes, 'little')
    print(cip_bytes_int)
    assert cip_bytes_int == cip

    # test int extraction
    test_start = time.time()
    for i in range(100000):
        cip_int = enc.ciphertext()
        exp_int = enc.exponent
    test_time = time.time() - test_start
    print("int extraction cost {:.2f}".format(test_time))

    # test bytes extraction
    test_start = time.time()
    for i in range(100000):
        cip_bytes = enc.ciphertext().to_bytes(572, 'little')
        exp_int = enc.exponent
        cip_int = int.from_bytes(cip_bytes, 'little')
    test_time = time.time() - test_start
    print("bytes extraction cost {:.2f}".format(test_time))

    # test str extraction
    test_start = time.time()
    for i in range(100000):
        cip_str = str(enc.ciphertext())
        exp_str = str(enc.exponent)
        cip_int = int(cip_str)
    test_time = time.time() - test_start
    print("str extraction cost {:.2f}".format(test_time))

    # public_key, private_key = paillier.generate_paillier_keypair()
    # secret_number_list = [3.141592653, 300, 0.01]
    # encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]
    # #print(encrypted_number_list)
    #
    # encrypted_sum = sum(encrypted_number_list)
    # decrypted_sum = private_key.decrypt(encrypted_sum)
    # print("decrypted sum = {}".format(decrypted_sum))
