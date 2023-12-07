import random
from rsa import miller_rabin_is_prime, binpow

def generate_number(bit_count):
    res = 0
    step = 1
    while bit_count > 0:
        current = (random.randint(0, 1_000_000) & 1)
        if current == 1:
            res += step

        step <<= 1
        bit_count -= 1
    
    res = (res | 1)

    return res

def generate_prime_numbers(bit_count, number_count):
    numbers = []
    while len(numbers) != number_count:
        num = generate_number(bit_count)
        if miller_rabin_is_prime(num, bit_count):
            numbers.append(num)

    return numbers

def generate_g(p, bit_count):
    while True:
        g = generate_number(bit_count)
        if binpow(g, p-1, p) == 1:
            return g

if __name__ == '__main__':
    p = generate_prime_numbers(512, 1)[0]
    a, b = generate_number(128), generate_number(128)
    g = generate_g(p, 16)
    
    A = binpow(g, a, p)
    B = binpow(g, b, p)
    K1 = binpow(A, b, p)
    K2 = binpow(B, a, p)

    print(K1 == K2)