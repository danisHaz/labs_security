import random

def get_key(key_seq, text_len):
    S = list(range(256))
    j = 0
    res = []

    # Key Scheduling Algorithm (KSA)
    for i in range(256):
        j = (j + S[i] + key_seq[i % len(key_seq)]) % 256
        S[i], S[j] = S[j], S[i]

    # Pseudo-Random Generation Algorithm (PRGA)
    i = j = 0
    for _ in range(text_len):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        k = S[(S[i] + S[j]) % 256]
        res.append(k)
    
    return res

def rc4(key_seq, text):
    k = get_key(key_seq, len(text))
    res = []

    for i, char in zip(range(len(text)), text):
        res.append(chr(ord(char) ^ k[i]))

    return ''.join(res)

def generate_key_sequence(key_len):
    return random.sample(list(range(0, 256))*5, key_len)

if __name__ == '__main__':
    key = generate_key_sequence(1)
    plaintext = "Hello, World!"
    ciphertext = rc4(key, plaintext)
    print("Ciphertext:", ciphertext)
    deciphertext = rc4(key, ciphertext)
    print(deciphertext)