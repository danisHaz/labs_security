from collections import Counter

def encode_str_into_int(s: str) -> list[int]:
    return [ord(s[i]) for i in range(len(s))]

def decode_str(letters: list[int]) -> str:
    return ''.join([chr(letter) for letter in letters])

def get_most_frequent_letter_in_str(s: str) -> str:
    return Counter(s).most_common(1)[0][0]