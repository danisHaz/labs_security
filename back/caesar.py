from language import determine_lang, determine_lang_bounds, get_most_frequent_letter_by_lang, determine_lang_len
from encode_decode import encode_str_into_int, decode_str, get_most_frequent_letter_in_str

def require_caesar_key_rules(key: str):
    key = str(key)
    if '.' in key or ',' in key:
        raise ValueError("provided key for caesar cipher is not valid")

def caesar_cipher(input_str: str, key: str) -> str:
    current_lang = determine_lang(input_str)

    require_caesar_key_rules(key)
    key = int(key)

    encoded_str = encode_str_into_int(input_str)
    encoded_cyphered_str = []

    for c in encoded_str:
        bounds = determine_lang_bounds(current_lang, chr(c))
        encoded_cyphered_str.append(((c % bounds[0] + key) % (bounds[1] - bounds[0] + 1) + bounds[0]))

    cyphered_str = decode_str(encoded_cyphered_str)

    return cyphered_str

def caesar_decipher(ciphered_str: str, key: int) -> str:
    current_lang = determine_lang(ciphered_str)
    bounds = determine_lang_bounds(current_lang, ciphered_str[0])

    require_caesar_key_rules(key)
    key = int(key)

    decipher_key = bounds[1] - bounds[0] + 1 - (key % (bounds[1] - bounds[0] + 1))

    return caesar_cipher(ciphered_str, decipher_key)

def caesar_hack(ciphered_str: str) -> str:
    current_lang = determine_lang(ciphered_str)
    lang_len = determine_lang_len(current_lang)
    freq_letter = get_most_frequent_letter_by_lang(current_lang)
    cur_most_common_letter = get_most_frequent_letter_in_str(''.join(ciphered_str.split()))

    key = (ord(cur_most_common_letter) - ord(freq_letter)) % lang_len

    return caesar_decipher(ciphered_str, key)