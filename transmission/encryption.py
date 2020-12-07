from multiprocessing import Pool

def encryptor(dist_arr, public_key):
    with Pool() as pool:
        encr = pool.map(public_key.encrypt, dist_arr)
    return encr

def decryptor(dist_arr, private_key):
    with Pool() as pool:
        decr = pool.map(private_key.decrypt, dist_arr)
    return decr


