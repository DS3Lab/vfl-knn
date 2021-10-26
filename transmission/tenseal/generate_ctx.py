import time

import tenseal as ts


if __name__ == '__main__':

    print("====== CKKS =======")

    # coefficient modulus size
    # 8192: [60, 40, 40, 60]
    # 8192: [40, 21, 21, 21, 21, 21, 21, 40]
    # 16384: [60, 40, 40, 40, 40, 40, 40, 40, 60]
    # 32768: [60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    ctx_file = "ts_ckks.config"
    context_bytes = context.serialize(save_secret_key=True)
    f = open(ctx_file, "wb")
    f.write(context_bytes)

    n_item = 1000
    v1 = range(n_item)
    v2 = range(n_item)
    n_iter = 100

    enc_start = time.time()
    for i in range(n_iter):
        enc_v1 = ts.ckks_vector(context, v1)
    enc_end = time.time()
    print("{} encryption of {} items cost {} s".format(n_iter, n_item, enc_end - enc_start))

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    sum_start = time.time()
    for i in range(n_iter):
        enc_sum = enc_v1 + enc_v2
    sum_end = time.time()
    print("{} sum of {} items cost {} s".format(n_iter, n_item, sum_end - sum_start))

    dec_start = time.time()
    for i in range(n_iter):
        dec_v1 = enc_v1.decrypt()
    dec_end = time.time()
    print("{} decryption of {} items cost {} s".format(n_iter, n_item, dec_end - dec_start))

    print("====== BFV =======")

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193
    )

    n_item = 1000
    v1 = range(n_item)
    v2 = range(n_item)
    n_iter = 100

    enc_start = time.time()
    for i in range(n_iter):
        enc_v1 = ts.bfv_vector(context, v1)
    enc_end = time.time()
    print("{} encryption of {} items cost {} s".format(n_iter, n_item, enc_end - enc_start))

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    sum_start = time.time()
    for i in range(n_iter):
        enc_sum = enc_v1 + enc_v2
    sum_end = time.time()
    print("{} sum of {} items cost {} s".format(n_iter, n_item, sum_end - sum_start))

    dec_start = time.time()
    for i in range(n_iter):
        dec_v1 = enc_v1.decrypt()
    dec_end = time.time()
    print("{} decryption of {} items cost {} s".format(n_iter, n_item, dec_end - dec_start))



