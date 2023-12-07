from itertools import zip_longest, cycle
from collections import Counter

class Answer:

    def __init__(self, answer: str = None, error: str = None) -> None:
        self.answer = answer
        self.error = error

en_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

ru_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890'

def findPos(s: str, alphabet: str):
    for pos in range(len(alphabet)):
        if alphabet[pos] == s:
            return pos
    
    return -1

def require_vizhener_key_rules(s: str, key: str) -> Answer:
    if len(s) < len(key):
        return Answer(error='Длина ключа больше длины шифруемого сообщения')

def vizhener_cipher(s: str, key: str, lang: str) -> Answer:
    cleared_s = ''
    alphabet = 0

    s = s.upper()

    print(s, key, lang)

    for letter in s:
        if lang == 'ru':
            alphabet = ru_alphabet
            if letter in ru_alphabet:
                cleared_s += letter
        elif lang == 'en':
            alphabet = en_alphabet
            if letter in en_alphabet:
                cleared_s += letter

    print(cleared_s)

    requirement = require_vizhener_key_rules(cleared_s, key)
    if requirement is not None:
        return requirement

    result = ''
    for letters in zip(cleared_s, cycle(key)):
        print(letters)
        result += alphabet[(findPos(letters[0], alphabet) + findPos(letters[1], alphabet)) % len(alphabet)]
    
    return Answer(answer = result)

def vizhener_decipher(s: str, key: str, lang: str) -> Answer:
    cleared_s = ''
    alphabet = 0

    for letter in s:
        if lang == 'ru':
            alphabet = ru_alphabet
            if letter in ru_alphabet:
                cleared_s += letter
        elif lang == 'en':
            alphabet = en_alphabet
            if letter in en_alphabet:
                cleared_s += letter

    requirement = require_vizhener_key_rules(cleared_s, key)
    if requirement is not None:
        return requirement

    result = ''
    for letters in zip(cleared_s, cycle(key)):
        result += alphabet[(findPos(letters[0], alphabet) + len(alphabet) - findPos(letters[1], alphabet)) % len(alphabet)]
    
    return Answer(answer = result)


def calculate_familiarity_indices(s: str, lang: str):
    counter = dict()
    for c in s:
        if c in counter:
            counter[c] += 1
        else:
            counter[c] = 1

    res = 0
    for (key, value) in counter.items():
        if value > 1:
            res += (value * (value - 1))

    return res / (len(s) * (len(s) - 1) if len(s) > 1 else 1)

def vizhener_hack(s: str, lang: str) -> tuple[str, Answer]:
    alph = en_alphabet if lang == 'en' else ru_alphabet
    s = ''.join(list(filter(lambda ch: ch in alph, s.upper())))
    print(s)

    step = 0.067 if lang == 'en' else 0.053
    real_length = -1
    familiarity_indices = []
    for possible_key_length in range(1, len(s)):
        res = 0
        for current_str in [s[i::possible_key_length] for i in range(possible_key_length)]:
            res += calculate_familiarity_indices(current_str, lang)

        res = res / possible_key_length

        familiarity_indices.append((possible_key_length, res))

        if abs(res - step) < 0.0125:
            print(res)
            real_length = possible_key_length
            break

    possible_key = ''

    for ind in range(real_length):
        sub_str = s[ind::real_length]
        most_common_letter = Counter(sub_str).most_common(1)[0][0]

        current_pos = 0
        if lang == 'en':
            current_pos = (findPos(most_common_letter, en_alphabet) - findPos('E', en_alphabet)) % len(en_alphabet)
            possible_key += en_alphabet[current_pos]
        else:
            current_pos = (findPos(most_common_letter, ru_alphabet) - findPos('О', ru_alphabet)) % len(ru_alphabet)
            possible_key += ru_alphabet[current_pos]

    return (possible_key, vizhener_decipher(s, possible_key, lang))