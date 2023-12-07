from flask import Flask, request
import json
import caesar
import vizhener
import binary_cipher
import rsa

app = Flask(__name__, template_folder='../front', static_folder='../front')
app.config.from_object({
    'CSRF_ENABLED': True,
    'CORS_HEADERS': 'Content-Type'
})

@app.route('/caesar_cipher', methods=['POST'])
def caesar_cipher():
    data_json = json.loads(request.get_data())
    key = data_json['key']
    text = data_json['text']

    status = 1
    ciphered_text = None

    try:
        ciphered_text = caesar.caesar_cipher(text, key)
    except:
        status = 0

    return json.dumps({
        "status": status,
        "data": ciphered_text,
    })

@app.route('/caesar_decipher', methods=['POST'])
def caesar_decipher():
    data_json = json.loads(request.get_data())
    key = data_json['key']
    text = data_json['text']

    status = 1
    deciphered_text = None

    try:
        deciphered_text = caesar.caesar_decipher(text, key)
    except:
        status = 0

    return json.dumps({
        "status": status,
        "data": deciphered_text,
    })

@app.route('/caesar_hack', methods=['POST'])
def caesar_hack():
    data_json = json.loads(request.get_data())
    text = data_json['text']

    status = 1
    hacked_text = None

    try:
        hacked_text = caesar.caesar_hack(text)
    except:
        status = 0

    return json.dumps({
        "status": status,
        "data": hacked_text,
    })

@app.route('/vizhener_cipher', methods=['POST'])
def vizhener_cipher():
    data_json = json.loads(request.get_data())
    text = data_json['text']
    key = data_json['key']
    lang = data_json['lang']

    status = 1
    ciphered_text = None

    try:
        ciphered_text = vizhener.vizhener_cipher(text, key, lang)
    except:
        status = 0

    if ciphered_text.error is not None:
        return json.dumps({
            "status": 0,
            "data": ciphered_text.error,
        })
    else:
        return json.dumps({
            "status": status,
            "data": ciphered_text.answer,
        })

@app.route('/vizhener_decipher', methods=['POST'])
def vizhener_decipher():
    data_json = json.loads(request.get_data())
    text = data_json['text']
    key = data_json['key']
    lang = data_json['lang']

    status = 1
    ciphered_text = None

    try:
        ciphered_text = vizhener.vizhener_decipher(text, key, lang)
    except:
        status = 0

    if ciphered_text.error is not None:
        return json.dumps({
            "status": 0,
            "data": ciphered_text.error,
        })
    else:
        return json.dumps({
            "status": status,
            "data": ciphered_text.answer,
        })

@app.route('/vizhener_hack', methods=['POST'])
def vizhener_hack():
    data_json = json.loads(request.get_data())
    text = data_json['text']
    lang = data_json['lang']

    status = 1
    ciphered_text = None

    # try:
    key, ciphered_text = vizhener.vizhener_hack(text, lang)
    # except:
    #     status = 0

    if ciphered_text.error is not None:
        return json.dumps({
            "status": 0,
            "data": ciphered_text.error,
        })
    else:
        return json.dumps({
            "status": status,
            "data": ciphered_text.answer,
            "key": key
        })

@app.route('/binary_cipher', methods=['POST'])
def binary_cipher_f():
    data_json = json.loads(request.get_data())
    text = data_json['text']
    key = data_json['key']

    status = 1
    ciphered_text = None

    ciphered_text = binary_cipher.cipher_binary(text, key)

    print(ciphered_text)

    return json.dumps({
        "status": status,
        "data": ciphered_text,
    })

@app.route('/binary_decipher', methods=['POST'])
def binary_decipher_f():
    data_json = json.loads(request.get_data())
    key = data_json['key']

    status = 1
    ciphered_text = None

    ciphered_text = binary_cipher.decipher_binary(key)

    print(ciphered_text)

    return json.dumps({
        "status": status,
        "data": ciphered_text,
    })

@app.route('/binary_generate', methods=['POST'])
def binary_generate():
    data_json = json.loads(request.get_data())
    keyLen = data_json['keyLen']

    status = 1
    key = binary_cipher.generate_binary_key(keyLen)

    print(key)

    return json.dumps({
        "status": status,
        "data": key,
    })

@app.route('/rsa_cipher', methods=['POST'])
def rsa_cipher_f():
    print('kek')
    data_json = json.loads(request.get_data())
    e = int(data_json['e'])
    N = int(data_json['N'])
    bit_count = int(data_json['bit_count'])
    text = data_json['text']

    status = 1
    ciphered_text = None

    ciphered_text = rsa.rsa_encode(text, e, N, bit_count)

    print(ciphered_text)

    return json.dumps({
        "status": status,
        "data": str(ciphered_text),
    })


@app.route('/rsa_decipher', methods=['POST'])
def rsa_decipher_f():
    data_json = json.loads(request.get_data())
    d = int(data_json['d'])
    N = int(data_json['N'])
    bit_count = int(data_json['bit_count'])
    text = int(data_json['ciphered'])

    status = 1
    ciphered_text = None

    ciphered_text = rsa.rsa_decode(text, d, N)

    print(ciphered_text)

    return json.dumps({
        "status": status,
        "data": ciphered_text,
    })

@app.route('/rsa_generate', methods=['POST'])
def rsa_generate_f():
    data_json = json.loads(request.get_data())
    bit_count = int(data_json['bit_count'])

    status = 1

    p, q, N, phi, e, d = rsa.generate_key(bit_count)

    print(p, q)

    return json.dumps({
        "status": status,
        "p": str(p),
        "q": str(q),
        "N": str(N),
        "phi": str(phi),
        "e": str(e),
        "d": str(d),
    })

@app.route('/rsa_hack', methods=['POST'])
def rsa_hack_f():
    data_json = json.loads(request.get_data())
    ciphered = int(data_json['ciphered'])
    e = int(data_json['e'])
    N = int(data_json['N'])

    d, deciphered = rsa.rsa_hack(ciphered, e, N)

    return json.dumps({
        "status": 1,
        "d": d,
        "deciphered": deciphered
    })

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = "5500")