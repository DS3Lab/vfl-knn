

def encryptor(dist_arr, public_key):
    return list(map(lambda x: public_key.encrypt(x), dist_arr))


def decryptor(dist_arr, private_key):
    return list(map(lambda x: private_key.decrypt(x), dist_arr))
