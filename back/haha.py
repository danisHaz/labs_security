from math import sqrt
from rsa import miller_rabin_is_prime, binpow
from itertools import chain
import numpy as np

# алго евклида из прошлой лабы
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

# бинпоиском ищем число x, что х*x == n
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

# решето Эратосфена, генерирует список простых чисел, которые <= n
# например, при n = 10 результат будет
# [2, 3, 5, 7]
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

# символ Лежандра
def legendre(a, p):
    return binpow(a, (p - 1) // 2, p)
 
# функция для решения уравнения (x*x) % p == N % p
# алгоритм Тонелли-Шэнкса
def tonelli(n, p):
    assert legendre(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        r = binpow(n, (p + 1) // 4, p)
        return r,p-r
    for z in range(2, p):
        if p - 1 == legendre(z, p):
            break
    c = binpow(z, q, p)
    r = binpow(n, (q + 1) // 2, p)
    t = binpow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = binpow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i

    return (r,p-r)

# генерирует факторную базу из B-гладких чисел
def find_base(N,B):

    factor_base = []
    primes = prime_gen(B)
    
    for p in primes:
        if legendre(N,p) == 1:
            # в факторную базу заносятся только те простые числа, которые
            # являются квадратичным вычетом по модулю p
            factor_base.append(p)
    return factor_base

# поиск B-гладких чисел
def find_smooth1(factor_base,N,I):

    # генерируем массив из чисел (x^2 - n) на отрезке [sqrt(N) - I, sqrt(N) + I]
    def sieve_prep(N,I):
        root = int(sqrt(N))
        return list(x**2 - N for x in range(root-I,root+I))
    
    sieve_seq = sieve_prep(N,I)
    sieve_list = sieve_seq.copy()
    
    factor_base_begin_ind = 0
    # если первое простое число - 2, то делаем отдельно
    if factor_base[0] == 2:
        factor_base_begin_ind = 1

        # делим каждое число на наше простое, пока делится
        for j in range(0,len(sieve_list)):
            while sieve_list[j] % 2 == 0:
                sieve_list[j] //= 2
        

    root = int(sqrt(N))
    for p in factor_base[factor_base_begin_ind:]:

        # находим два числа x1 и x2, которые подходят под
        # (x*x) % p == N % p 
        residues = tonelli(N,p)

        for r in residues:
            # просто делим каждое число на простой делитель, пока делится
            for i in range(len(sieve_list)):
                while sieve_list[i] % p == 0:
                    sieve_list[i] //= p
                    
            # идем с конца списка и делаем то же самое
            # for i in range(((r-root+I) % p)+I, 0, -p):
            #     while sieve_list[i] % p == 0:
            #         sieve_list[i] //= p
                    
    indices = [] # index of discovery
    xlist = [] #original x terms
    smooth_nums = []
    
    print(sieve_list)

    # теперь sieve_list будет содержать только единицы и те числа, простые делители которых
    # больше заданного B (то есть эти числа не будут B-гладкими)
    # значит, их мы убираем, а вот числа, которые превратились в 1, добавляем в smooth_nums
    for i in range(len(sieve_list)):
        if len(smooth_nums) >= len(factor_base)+1:
            break
        elif sieve_list[i] == 1 or sieve_list[i] == -1: # found B-smooth number
            smooth_nums.append(sieve_seq[i])
            xlist.append(i+root-I)
            indices.append(i)
    return smooth_nums, xlist, indices

# строим матрицу из единиц и нулей (как в примере в методичке)
def build_matrix(smooth_nums,factor_base):

    # функция для факторизации числа, то есть находит список всех простых делителей
    # из факторной базы
    # например, для n = 12 результат будет
    # [2, 2, 3]
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

    # результирующая матрица, которая будет из 0 и 1
    M = []
    # расширяем факторную базу на всякий случай
    factor_base.insert(0,-1)

    for n in smooth_nums:
        # вектор, в котором будут записаны количества простых делителей n
        # например, если n = 12 и факторная база [2, 3, 5, 7]
        # то exp_vector = [0, 1, 0, 0]
        # то есть [2 % 2, 1 % 2, 0 % 2, 0 % 2]
        # поскольку 12 дважды делится на 2 и один раз делится на 3
        exp_vector = [0]*(len(factor_base))
        n_factors = factor(n,factor_base)
        for i in range(len(factor_base)):
            if factor_base[i] in n_factors:
                exp_vector[i] = (exp_vector[i] + n_factors.count(factor_base[i])) % 2

        # если нашли какую-то строку только из нулей
        # значит, наше гладкое число подходит
        if 1 not in exp_vector:
            return True, n
        else:
            pass
        
        M.append(exp_vector)

    return (False, transpose(M))

# транспонирует матрицу
def transpose(matrix):
    matrix = np.array(matrix)
    return matrix.T

# метод Гаусса
def gauss_elim(M):
    marks = np.zeros(M.shape[0])
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            num = M[i, j]
            row = M[i, :]
            if num == 1:
                marks[j] = 1
                for k in chain(range(0,i),range(i+1,M.shape[0])): #search for other 1s in the same column
                    if M[k, j] == 1:
                        for i in range(M.shape[1]):
                            M[k, i] = (M[k, i] + row[i]) % 2
                break
    M = transpose(M)
    
    sol_rows = []
    for i in range(len(marks)): #find free columns (which have now become rows)
        if marks[i] == 0:
            free_row = [M[i, :], i]
            sol_rows.append(free_row)
    
    if not sol_rows:
        return "No solution found. Need more smooth numbers."

    return sol_rows, marks, M

# пытаемся превратить выбранную строку матрицы M в строку из нулей
# через линейные преобразования
def solve_row(sol_rows, M, marks, K=0):
    solution_vec, indices = [],[]
    free_row = sol_rows[K][0] # may be multiple K
    for i in range(free_row.shape[0]):
        if free_row[i] == 1:
            indices.append(i)
    
    for r in range(M.shape[0]): #rows with 1 in the same column will be dependent
        if not marks[r]:
            continue

        for i in indices:
            if M[r, i] == 1:
                solution_vec.append(r)
                break

    solution_vec.append(sol_rows[K][1])       
    return np.array(solution_vec)

# тут считаем произведение x_i, которые мы предварительно выбрали
# и произведение чисел (x_i*x_i - N)
# получается, что последнее прозведение имеет (с какой-то вероятностью)
# нетривиальный общий делитель
# значит, можем через алгоритм Евклида найти НОД - это будет один из делителей
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

# функция, раскладывающая N на p и q
# n - число N
# B - максимальный простой делитель у x_i, которые мы будем перебирать
# I - будем перебирать x_i в интервале [sqrt(N) - I; sqrt(N) + I]
def find_factors(N, B, I):
    # если простое, то разложить не получится
    if miller_rabin_is_prime(N):
        return (None, None)
    
    # пытаемся найти такое число х, что х*х == N
    real_sqrt = isqrt(N)
    if real_sqrt is not None:
        # если такое число нашлось, то число N = x*x
        return (real_sqrt, real_sqrt)
    
    # находим факторную базу
    factor_base = find_base(N,B)
    
    # вычисляем B-гладкие числа
    smooth_nums, xlist, indices = find_smooth1(factor_base, N,I)
    
    print(factor_base, smooth_nums)

    if len(smooth_nums) < len(factor_base):
        return ("Мало B-гладких чисел, необходимо увеличить параметр B.", None)
    
    # строим матрицу из 1 и 0
    is_square, t_matrix = build_matrix(smooth_nums,factor_base)

    # если сразу нашли нулевую строку, то нашли и делители
    if is_square == True:
        x = smooth_nums.index(t_matrix)
        factor, x1, y1 = euclid(xlist[x]+int(sqrt(t_matrix)),N)
        return abs(int(factor)), abs(int(N//factor))

    # иначе приводим эту матрицу к ступенчатому виду через метод Гаусса
    sol_rows,marks,M = gauss_elim(t_matrix)

    # затем берем первый вектор и пытаемся его привести к нулевому
    solution_vec = solve_row(sol_rows,M,marks,0)

    # вычисляем делитель числа N
    factor = solve(solution_vec,smooth_nums,xlist,N)

    for K in range(1,len(sol_rows)):
        # если на K-ой попытке снова нашли тривиальный делитель
        # то продолжаем искать
        if factor == 1 or factor == N:
            solution_vec = solve_row(sol_rows,M,marks,K)
            factor = solve(solution_vec,smooth_nums,xlist,N)
        else:
            # иначе выводим ответ
            return abs(int(factor)), abs(int(N//factor))

    # не получили ответ, плачем 0_0
    return (None, None)

def find_d(p, q, e):
    phi = (p - 1) * (q - 1)
    d, x, y = euclid(e, phi)

    assert(d == 1)

    x = (x % phi + phi) % phi
    return x

def find_bin(n):
    ans = 0
    while n > 0:
        n >>= 1
        ans += 1

    return ans

if __name__ == '__main__':
    # p, q = find_factors(344572667627327574872986520507, 10_000_000, 500)
    # print(p, q)
    # assert(p == 50539 or p == 49831)
    # assert(q == 50539 or q == 49831)

    p, q = find_factors(19908397, 100, 400)
    print(p, q)
    assert(p == 6719 or p == 2963)
    assert(q == 6719 or q == 2963)

    p, q = find_factors(34413479, 100, 400)
    print(p, q)
    assert(p == 1723 or p == 19973)
    assert(q == 1723 or q == 19973)

    p, q = find_factors(725841031, 100, 400)
    print(p, q)
    assert(p == 25469 or p == 28499)
    assert(q == 25469 or q == 28499)

    print("ALL OK!")