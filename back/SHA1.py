def rotate_left(n, b):
    return ((n << b) | (n >> (32 - b))) & 0xffffffff


def hash_block(block, h0, h1, h2, h3, h4):
    assert len(block) == 64

    w = [0] * 80

    # Break block into sixteen 4-byte big-endian words w[i]
    for i in range(16):
        w[i] = int.from_bytes(block[i * 4:i * 4 + 4], byteorder='big')

    # Extend the sixteen 4-byte words into eighty 4-byte words
    for i in range(16, 80):
        w[i] = rotate_left(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)

    # Initialize hash value for this block
    a = h0
    b = h1
    c = h2
    d = h3
    e = h4

    for i in range(80):
        if 0 <= i <= 19:
            # Use alternative 1 for f from FIPS PB 180-1 to avoid bitwise not
            f = d ^ (b & (c ^ d))
            k = 0x5A827999
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = 0x6ED9EBA1
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = 0x8F1BBCDC
        elif 60 <= i <= 79:
            f = b ^ c ^ d
            k = 0xCA62C1D6

        a, b, c, d, e = ((rotate_left(a, 5) + f + e + k + w[i]) & 0xffffffff,
                         a, rotate_left(b, 30), c, d)

    # Add this block's hash to result so far
    h0 = (h0 + a) & 0xffffffff
    h1 = (h1 + b) & 0xffffffff
    h2 = (h2 + c) & 0xffffffff
    h3 = (h3 + d) & 0xffffffff
    h4 = (h4 + e) & 0xffffffff

    return h0, h1, h2, h3, h4

def update(arg, unprocessed, message_byte_length, h):
    """Update the current digest.

    This may be called repeatedly, even after calling digest or hexdigest.

    Arguments:
        arg: bytes, bytearray, or BytesIO object to read from.
    """

    # Try to build a block out of the unprocessed data, if any
    reader_position = 64 - len(unprocessed)
    block = unprocessed + arg[0:reader_position]


    # Read the rest of the data, 64 bytes at a time
    while len(block) == 64:
        h = hash_block(block, *h)
        message_byte_length += 64
        block = arg[reader_position:reader_position + 64]
        reader_position += 64

    unprocessed = block

    return (unprocessed, message_byte_length, h)

def digest(unprocessed, message_byte_length, h):
    """Produce the final hash value (big-endian) as a bytes object"""
    res = 0
    step = 32 * 4 # сдвигаем каждый блок на 32 бита влево, чтобы соединить вместе
    for h in produce_digest(unprocessed, message_byte_length, h):
        res += (h << step)
        step -= 32
    return res

def hexdigest(unprocessed, message_byte_length, h):
    """Produce the final hash value (big-endian) as a hex string"""
    return '%08x%08x%08x%08x%08x' % produce_digest(unprocessed, message_byte_length, h)

def produce_digest(unprocessed, message_byte_length, h):
    """Return finalized digest variables for the data processed so far."""
    # Pre-processing:
    message = unprocessed
    message_byte_length = message_byte_length + len(message)

    # append the bit '1' to the message
    message += b'\x80'

    # append 0 <= k < 512 bits '0', so that the resulting message length (in bytes)
    # is congruent to 56 (mod 64)
    message += b'\x00' * ((56 - (message_byte_length + 1) % 64) % 64)

    # append length of message (before pre-processing), in bits, as 64-bit big-endian integer
    message_bit_length = message_byte_length * 8
    # message += struct.pack(b'>Q', message_bit_length)
    message += message_bit_length.to_bytes(8, byteorder='big')


    # Process the final block
    # At this point, the length of the message is either 64 or 128 bytes.
    h = hash_block(message[:64], *h)
    if len(message) == 64:
        return h
    return hash_block(message[64:], *h)


def sha1(data):
    h = (
        0x67452301,
        0xEFCDAB89,
        0x98BADCFE,
        0x10325476,
        0xC3D2E1F0,
    )

    unprocessed = b''
    message_byte_length = 0

    data = bytes(data, encoding='utf-8')

    unprocessed, message_byte_length, h = update(data, unprocessed, message_byte_length, h)
    return hexdigest(unprocessed, message_byte_length, h)


if __name__ == '__main__':
    result = sha1(input())
    # result = int.from_bytes(result, byteorder='big', signed=False)
    print(result)