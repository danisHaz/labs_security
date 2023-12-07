from functools import reduce
import numpy as np
import random

en_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
ru_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890'

def to_bits(s):
    result = []
    
    if type(s) is not bytes:
        s = bytes(s, 'utf-8')
    for number in [*map(lambda a: int(a), s)]:
        bitarr = []
        for _ in range(8):
            bitarr.append(f'{number & 1}')
            number >>= 1

        bitarr.reverse()
        result.append(bitarr)
    return reduce(lambda a, b: a + b, result)

def from_bits(s):
    int_arr = []
    for number in [s[i:i+8] for i in range(0, len(s), 8)]:
        res = 0
        step = 1
        for j in range(len(number)-1, -1, -1):
            res += (1 if number[j] == '1' else 0) * step
            step <<= 1

        int_arr.append(res)
    
    print(int_arr)
    return bytes.decode(bytes(int_arr), encoding='utf-8')

def xor(first: str, second: str):
    if len(first) != len(second):
        return -1
    
    res = ''
    for i in range(len(first)):
        if first[i] == second[i]:
            res += '0'
        else:
            res += '1'

    return res

def cipher_binary(s: str, key: str):
    binary_s = to_bits(bytes(s, 'utf-8'))
    binary_key = ''
    if len(key) > len(s):
        if len(binary_s) == len(key):
            binary_key = key
        else:
            binary_key = to_bits(bytes(key, 'utf-8'))
    else:
        binary_key = to_bits(bytes(key, 'utf-8'))

    print(binary_s, binary_key)

    if len(binary_s) > len(binary_key):
        binary_key = binary_key * (len(binary_s) // len(binary_key)) + binary_key[:(len(binary_s) % len(binary_key))]
    elif len(binary_s) < len(binary_key):
        binary_key = binary_key[:len(binary_s)]

    result = ''
    for i in range(0, len(binary_s), len(binary_key)):
        lenn = min(len(binary_s) - i, len(binary_key))
        res = xor(binary_key[0 : lenn], binary_s[i : i + lenn])
        result += res
        binary_key = res

    with open('kek.txt', 'w') as f:
        f.write(result)

    return result

def decipher_binary(key: str):
    s = ''
    with open('kek.txt', 'r') as f:
        s = f.read()

    s = s.strip()

    binary_key = ''
    # if len(s) < len(key):
    #     raise ValueError()
    binary_key = to_bits(bytes(key, 'utf-8'))

    if len(key) == len(s):
        binary_key = key
    elif len(binary_key) == len(s):
        pass
    else:
        binary_key = to_bits(bytes(key, 'utf-8'))

    if len(s) > len(binary_key):
        binary_key = binary_key * (len(s) // len(binary_key)) + binary_key[:(len(s) % len(binary_key))]
    elif len(s) < len(binary_key):
        binary_key = binary_key[:len(s)]

    result = ''
    for i in range(0, len(s), len(binary_key)):
        lenn = min(len(s) - i, len(binary_key))
        res = xor(binary_key[0 : lenn], s[i : i + lenn])
        result += res
        binary_key = s[i : i + lenn]
    print(result)
    result = from_bits(result)

    return result

# генерация ключа
def generate_binary_key(s: str): # s - строка для шифрования

    # смотрим на длину бинарной строки, полученной из текста
    binary_s_len = len(to_bits(bytes(s, 'utf-8')))

    # генерируем позиции, на которых будут нули
    # генерируем их ровно половину от длины исходной бинарной строки
    zeros_positions = random.sample(range(0, binary_s_len), binary_s_len // 2)

    key = ''
    for i in range(binary_s_len):
        # если текущий индекс есть среди позиций нулей, то кладем туда ноль
        if i in zeros_positions:
            key += '0'
        else:
            key += '1' # иначе единицу

    return key

if __name__ == '__main__':
    s = 'qweuiq'
    key = 'jafoei'
    ciphered = cipher_binary(s, key)
    print(ciphered)
    deciphered = decipher_binary(ciphered, key)
    print(deciphered)