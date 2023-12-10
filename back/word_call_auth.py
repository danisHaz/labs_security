from SHA1 import sha1
from random import random, sample
from rsa import rsa_encode, rsa_decode, generate_key

def generate_secret_word(word_len=64):
    secret = ''.join([
        chr(i) for i in sample(
            list(range(50, 130)) * 5,
            word_len
        )
    ])
    return secret

def sha1_combined(password, hashed_secret_word):
    return sha1(sha1(password) + hashed_secret_word)

def sign_data(data, private_key, N, bit_count):
    hashed_data = sha1(data)
    encoded_hash = rsa_encode(hashed_data, private_key, N, bit_count)
    return (data, encoded_hash)

def unsign_data(data, encoded_hash, public_key, N):
    decoded_hash = rsa_decode(encoded_hash, public_key, N)
    hashed_data = sha1(data)
    return (hashed_data, decoded_hash)

if __name__ == '__main__':
    secret = generate_secret_word()
    password = 'super secret password'
    h = sha1_combined(password, sha1(secret))
    print(secret, password, sha1_combined(password, secret), sep = '\n')
    p, q, N, phi, e, d = generate_key(512)
    print('e', e, 'N', N, 'd', d)
    data, encoded_hash = sign_data(h, d, N, 512)
    print(data, encoded_hash)
    hashed_data, decoded_hash = unsign_data(data, encoded_hash, e, N)
    print(hashed_data, decoded_hash)
    if hashed_data == decoded_hash:
        print(True)
    else:
        print(False)