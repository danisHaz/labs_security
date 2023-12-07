class Dictionary_bound:
    def __init__(self, bound_ranges: list[tuple] | tuple) -> None:
        self.bound_ranges = bound_ranges

    def is_char_fits_bounds(self, ch: str) -> bool:
        for rng in self.bound_ranges:
            if rng[0] <= ch and ch <= rng[1]:
                return True
        
        return False
    
    def get_appropriate_bounds(self, ch: str) -> tuple:
        for rng in self.bound_ranges:
            if rng[0] <= ch and ch <= rng[1]:
                return rng

def __get_en_bounds() -> Dictionary_bound:
    return Dictionary_bound([('a', 'z'), ('A', 'Z'), (' ', ' '), ('0', '9')])

def __get_ru_bounds() -> Dictionary_bound:
    return Dictionary_bound([('а', 'я'), ('А', 'Я'), (' ', ' '), ('0', '9')])

def __is_str_fits_lang(s: str, lang: str) -> bool:
    lang_bounds = None
    if lang == 'en':
        lang_bounds = __get_en_bounds()
    elif lang == 'ru':
        lang_bounds = __get_ru_bounds()
    else:
        return False
    
    for c in s:
        if not lang_bounds.is_char_fits_bounds(c):
            return False
    
    return True

def determine_lang(s: str) -> str:
    if __is_str_fits_lang(s, 'en'):
        return 'en'
    elif __is_str_fits_lang(s, 'ru'):
        return 'ru'
    else:
        return 'indeterminate'

def determine_lang_len(s: str) -> str:
    if s == 'en':
        return 26
    elif s == 'ru':
        return 33
    else:
        return -1

def determine_lang_bounds(lang: str, ch: str) -> Dictionary_bound:
    bounds = None
    if lang == 'en':
        bounds = __get_en_bounds()
    elif lang == 'ru':
        bounds = __get_ru_bounds()
    else: # suppose we have indeterminate language
        raise ValueError('Cannot get dictionary bounds for indeterminate language')
    
    real_bounds = bounds.get_appropriate_bounds(ch)
    return (ord(real_bounds[0]), ord(real_bounds[1]))

def get_most_frequent_letter_by_lang(lang: str) -> str:
    if lang == 'en':
        return 'e'
    elif lang == 'ru':
        return 'о'
    else:
        raise ValueError('Cannot get most frequent letter')