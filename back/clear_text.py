s = ''
st = ''
while st != '0':
    st = input()
    s += st

s = s.upper()
mp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
s = ''.join(list((filter(lambda c: c in mp, s))))

print(s)