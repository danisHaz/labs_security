import random
from functools import reduce
from math import sqrt
from itertools import chain

# переделываем исходный текст в строку битов
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

def int_to_bits(n, bit_count):
    res = []
    while n != 0:
        if (n & 1) == 1:
            res.append('1')
        else:
            res.append('0')
        n >>= 1

    for i in range(len(res) - bit_count):
        res.append('0')

    res.reverse()
    return reduce(lambda a, b: a + b, res)

def find_max_power(a):

    res = 0
    while a > 0:
        a >>= 1
        res += 1

    return res


def to_ints(s: list, bit_count):
    s.reverse()
    result = []
    for number_str in [s[i : i + bit_count] for i in range(0, len(s), bit_count)]:
        res = 0
        step = 1
        for number in number_str:
            if number == '1':
                res += step
            
            step <<= 1
        result.append(res)
    return reduce(lambda a, b: a + b, result)

# def to_int(s: str):


def from_bits(s):
    int_arr = []
    for number in [s[i:i+8] for i in range(0, len(s), 8)]:
        res = 0
        step = 1
        for j in range(len(number)-1, -1, -1):
            res += (1 if number[j] == '1' else 0) * step
            step <<= 1

        int_arr.append(res)
    
    return bytes.decode(bytes(int_arr), encoding='utf-8')

def binpow(a, n, mod):

    res = 1
    while n > 0:
        if n % 2 != 0:
            res = (res * a) % mod
        a = (a * a) % mod
        n //= 2

    return res

def inner_miller(n, d):
    a = random.randint(2, n - 2)
    x = binpow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    
    while d != n - 1:
        x = (x * x) % n
        d *= 2

        if x == 1: return False
        if x == n - 1: return True

    return False

def miller_rabin_is_prime(num, repeat = 10):
    if num == 2 or num == 3:
        return True
    
    if (num & 1) == 0:
        return False
    
    orig_num = num
    num = num - 1
    r = 0
    d = -1
    while num > 0:
        if (num & 1) == 0:
            d = num
            break
        num >>= 1
        r += 1

    for t in range(repeat):
        if not inner_miller(orig_num, d):
            return False
        
    return True

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

def generate_p_q(bit_count):
    pq = []
    while len(pq) != 2:
        num = generate_number(bit_count)
        if miller_rabin_is_prime(num, bit_count):
            pq.append(num)

    return pq

# расширенный алго евклида
# возвращает НОД(a, b), и коэффициенты x, y
def euclid(a, b):
    ab = [(a, b)]
    while a != 0:
        a, b = b % a, a
        ab.append((a, b))

    xy = [(0, 1)]
    gcd = b
    ind = len(ab) - 2
    while ind != -1:
        xy.append((xy[-1][1] - (ab[ind][1] // ab[ind][0]) * xy[-1][0], xy[-1][0]))
        ind -= 1

    return gcd, xy[-1][0], xy[-1][1]

def generate_key(bit_count):
    seed = 0
    with open('rsa.txt', 'r') as f:
        seed = int(f.read())

    random.seed(seed)
    while True:
        p, q = generate_p_q(bit_count)
        N = p * q
        phi = (p - 1) * (q - 1)
        e = generate_number(bit_count * 2 // 3)
        d, x, y = euclid(e, phi)
        if d != 1:
            continue
        x = (x % phi + phi) % phi

        with open('rsa.txt', 'w') as f:
            f.write(str(N))

        return (p, q, N, phi, e, x)

def rsa_encode(s, e, N, bit_count):
    s_bytes = to_ints(to_bits(s), bit_count)

    # ciphered = to_ints(reduce(lambda a,b: a + b, list(map(lambda st: int_to_bits(binpow(st, e, N), bit_count), s_bytes))), len(s_bytes) * bit_count)[0]
    ciphered = binpow(s_bytes, e, N)

    return ciphered

def rsa_decode(ciphered, d, N):
    deciphered = binpow(ciphered, d, N)

    bits_len = find_max_power(deciphered)

    s_bytes = int.to_bytes(deciphered, length=((bits_len + 7) // 8), byteorder='big')

    res = 'Не удалось расшифровать'
    try:
        res = (bytes.decode(s_bytes, encoding='utf-8'))
    except:
        pass
    finally:
        return res

def rsa_hack(ciphered, e, N):
    print(N)
    p, q = find_factors(N, 20000, 20000)
    if p is None or type(p) == type(''):
        print(p)
        return None, None

    d = find_d(p, q, e)
    deciphered = rsa_decode(ciphered, d, N)

    print(p, q, d)

    return d, deciphered


def euclid(a, b):
    ab = [(a, b)]
    while a != 0:
        a, b = b % a, a
        ab.append((a, b))

    xy = [(0, 1)]
    gcd = b
    ind = len(ab) - 2
    while ind != -1:
        xy.append((xy[-1][1] - (ab[ind][1] // ab[ind][0]) * xy[-1][0], xy[-1][0]))
        ind -= 1

    return gcd, xy[-1][0], xy[-1][1]

def isqrt(n):
    l, r = 1, n
    while r - l > 1:
        tm = (l + r) // 2
        if tm * tm > n:
            r = tm
        else:
            l = tm

    if l * l == n:
        return l
    if r * r == n:
        return r
    
    return None

def prime_gen(n):
    if n < 2:
        return list()

    isPrime = list(True for i in range(n+1))

    isPrime[0]=False
    isPrime[1]=False

    for j in range(2, int(n/2)):
        if isPrime[j]:
            for i in range(2*j, n+1, j):
                isPrime[i] = False

    primes = list()
    for i in range(0, n+1):
        if isPrime[i]:
            primes.append(i)
            
    return primes

def legendre(a, p):
    return binpow(a, (p - 1) // 2, p)

def find_base(N,B):

    factor_base = []
    primes = prime_gen(B)
    
    for p in primes:
        if legendre(N,p) == 1:
            factor_base.append(p)
    return factor_base

def find_smooth1(factor_base,N,I):
    def sieve_prep(N,I):
        root = int(sqrt(N))
        return list(x**2 - N for x in range(root-I,root+I))
    
    sieve_seq = sieve_prep(N,I)
    sieve_list = sieve_seq.copy()
    
    factor_base_begin_ind = 0

    if factor_base[0] == 2:
        factor_base_begin_ind = 1
        for j in range(0,len(sieve_list)):
            while sieve_list[j] % 2 == 0:
                sieve_list[j] //= 2
        
    root = int(sqrt(N))
    for p in factor_base[factor_base_begin_ind:]:
        print(p)
        for i in range(len(sieve_list)):
            while sieve_list[i] % p == 0:
                sieve_list[i] //= p

    indices = []
    xlist = []
    smooth_nums = []
    
    for i in range(len(sieve_list)):
        print(i)
        if len(smooth_nums) >= len(factor_base)+1:
            break
        elif sieve_list[i] == 1 or sieve_list[i] == -1:
            smooth_nums.append(sieve_seq[i])
            xlist.append(i+root-I)
            indices.append(i)
    return smooth_nums, xlist, indices

def build_matrix(smooth_nums,factor_base):

    def factor(n,factor_base):
        factors = []
        if n < 0:
            factors.append(-1)
        for p in factor_base:
            if p == -1:
                pass
            else:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        return factors


    M = []

    factor_base.insert(0,-1)

    for n in smooth_nums:
        exp_vector = [0]*(len(factor_base))
        n_factors = factor(n,factor_base)
        for i in range(len(factor_base)):
            if factor_base[i] in n_factors:
                exp_vector[i] = (exp_vector[i] + n_factors.count(factor_base[i])) % 2

        if 1 not in exp_vector:
            return True, n
        else:
            pass
        
        M.append(exp_vector)

    return (False, transpose(M))

def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        new_matrix.append(new_row)
    return(new_matrix)

def gauss_elim(M):
    marks = [False]*len(M[0])
    
    for i in range(len(M)):
        for j in range(len(M[i])):
            num = M[i][j]
            row = M[i]
            if num == 1:
                marks[j] = True
                for k in chain(range(0,i),range(i+1,len(M))):
                    if M[k][j] == 1:
                        for i in range(len(M[k])):
                            M[k][i] = (M[k][i] + row[i])%2
                break
    M = transpose(M)

    sol_rows = []
    for i in range(len(marks)):
        if not marks[i]:
            free_row = [M[i],i]
            sol_rows.append(free_row)

    if not sol_rows:
        return "No solution found. Need more smooth numbers."

    return sol_rows, marks, M

def solve_row(sol_rows, M, marks, K=0):
    solution_vec, indices = [],[]
    free_row = sol_rows[K][0]
    for i in range(len(free_row)):
        if free_row[i] == 1:
            indices.append(i)
    
    for r in range(len(M)):
        if not marks[r]:
            continue

        for i in indices:
            if M[r][i] == 1:
                solution_vec.append(r)
                break

    solution_vec.append(sol_rows[K][1])       
    return solution_vec

def solve(solution_vec, smooth_nums, xlist, N):
    
    solution_nums = [smooth_nums[i] for i in solution_vec]
    x_nums = [xlist[i] for i in solution_vec]
    # print(solution_nums,x_nums)
    
    Asquare = 1
    for n in solution_nums:
        Asquare *= n
        
    b = 1
    for n in x_nums:
        b *= n

    a = isqrt(Asquare)    
    factor, x1, y1 = euclid(b-a,N)
    return factor

def find_factors(N, B, I):

    if miller_rabin_is_prime(N):
        return (None, None)
    

    real_sqrt = isqrt(N)
    if real_sqrt is not None:
    
        return (real_sqrt, real_sqrt)
    

    factor_base = find_base(N,B)

    print('base found')
    
    smooth_nums, xlist, indices = find_smooth1(factor_base, N,I)

    print('smooth found')
    
    if len(smooth_nums) < len(factor_base):
        return ("Мало B-гладких чисел, необходимо увеличить параметр B.", None)
    

    is_square, t_matrix = build_matrix(smooth_nums,factor_base)

    print('matrix build')

    if is_square == True:
        x = smooth_nums.index(t_matrix)
        factor, x1, y1 = euclid(xlist[x]+int(sqrt(t_matrix)),N)
        return abs(int(factor)), abs(int(N//factor))


    sol_rows,marks,M = gauss_elim(t_matrix)


    print('gauss eliminated')

    solution_vec = solve_row(sol_rows,M,marks,0)


    print('row solved')

    factor = solve(solution_vec,smooth_nums,xlist,N)

    for K in range(1,len(sol_rows)):
    
    
        if factor == 1 or factor == N:
            solution_vec = solve_row(sol_rows, M, marks, K)
            factor = solve(solution_vec,smooth_nums,xlist,N)
        else:
        
            return abs(int(factor)), abs(int(N//factor))


    return (None, None)

def find_d(p, q, e):
    phi = (p - 1) * (q - 1)
    d, x, y = euclid(e, phi)

    assert(d == 1)

    x = (x % phi + phi) % phi
    return x

if __name__ == '__main__':

    # p, q = find_factors(344572667627327574872986520507, 10000000, 1000)
    # print(p, q)
    # assert(p == 50539 or p == 49831)
    # assert(q == 50539 or q == 49831)

    # res = generate_key(30)
    # print(res)
    # exit(0)

    # # 20 bit
    # p, q = find_factors(845932463539, 10_000, 10_000)
    # print(p, q)
    # assert(p == 1037329 or p == 815491)
    # assert(q == 1037329 or q == 815491)

    # p, q = find_factors(140210327261, 10_000, 10_000)
    # print(p, q)
    # assert(p == 515231 or p == 272131)
    # assert(q == 515231 or q == 272131)

    # p, q = find_factors(74069196391, 10_000, 10_000)
    # print(p, q)
    # assert(p == 541129 or p == 136879)
    # assert(q == 541129 or q == 136879)

    # # 22 bit
    p, q = find_factors(3261434502421303279323564402143, 1_000_000, 200_000)
    print(p, q)
    # assert(p == 2048779 or p == 539129)
    # assert(q == 2048779 or q == 539129)

    # p, q = find_factors(1857319253399, 10_000, 10_000)
    # print(p, q)
    # assert(p == 1268173 or p == 1464563)
    # assert(q == 1268173 or q == 1464563)

    # p, q = find_factors(4298387984801, 10_000, 10_000)
    # print(p, q)
    # assert(p == 3323137 or p == 1293473)
    # assert(q == 3323137 or q == 1293473)

    # 30 bit
    # p, q = find_factors(38256069088181281, 10_000, 10_000)
    # print(p, q)
    # assert(p == 158174659 or p == 241859659)
    # assert(q == 158174659 or q == 241859659)

    # p, q = find_factors(119002382626929623, 15_000, 15_000)
    # print(p, q)
    # assert(p == 450504409 or p == 264153647)
    # assert(q == 450504409 or q == 264153647)

    # p, q = find_factors(93000342617009329, 10_000, 10_000)
    # print(p, q)
    # assert(p == 268448107 or p == 346436947)
    # assert(q == 268448107 or q == 346436947)  

    # p, q = find_factors(344572667627327574872986520507, 500_000, 500_000)
    # print(p, q)
    # assert(p == 268448107 or p == 346436947)
    # assert(q == 268448107 or q == 346436947)

    print("ALL OK!")