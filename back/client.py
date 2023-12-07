import socket
import pickle
import threading
from SHA1 import sha1
from rsa import generate_key, binpow
from word_call_auth import sign_data
from diffie_hellmann import generate_number
from rc4 import rc4
import json
from flask import Flask, request

class Client:
    __max_timeout = 1e10

    def __init__(self, host='localhost', port=15001) -> None:
        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__host = host
        self.__port = port

        self.__diffie_key = None
        self.__diffie_B = None

    def connect(self):
        self.__connection.connect((self.__host, self.__port))
    
    def disconnect(self):
        self.__connection.close()
    
    def send(self, data):
        '''Send data via created socket connection'''
        buffer = pickle.dumps(data)
        self.__connection.send(buffer)
    
    def sendForResult(self, data, with_timeout: int or None = None) -> dict | None:
        '''Send data via socket connection and wait until response is coming'''
        self.send(data)
        # max_operation_count = with_timeout if with_timeout is not None else Client.__max_timeout
    
        ans_buffer = self.__connection.recv(1024)
        if len(ans_buffer) != 0:
            answer = pickle.loads(ans_buffer)
            return answer
        
        return None

    def _send_login(self, login: str) -> dict | None:
        request = {
            'type': 'auth',
            'login': login,
        }

        print('client: sending login initiated')

        result = self.sendForResult(request)
        return result

    def _check_hash(self, password, hashed_secret_word) -> dict | None:
        print('client checking password hash initiated')

        H = sha1(sha1(password) + hashed_secret_word)
        p, q, N, phi, e, d = generate_key(512)
        data, encoded_hash = sign_data(H, d, N, 512)

        print(f'client: H = {H}, N = {N}, e = {e}, d = {d}')
        print(f'client: data = {data}, encoded_hash={encoded_hash}')

        request = {
            'type': 'check_hash',
            'H': H,
            'data': data,
            'encoded_hash': encoded_hash,
            'public_key': e,
            'N': N,
        }

        result = self.sendForResult(request)
        return result

    def _send_session_key_final(self, g, p, bit_count=128) -> dict | None:
        self.__diffie_B = generate_number(bit_count)
        B = binpow(g, self.__diffie_B, p)

        print(f'client: diffie B = {B}')

        request = {
            'type': 'sess_keygen_final',
            'B': B
        }

        result = self.sendForResult(request)
        return result

    def send_message(self, message: str):
        encrypted_message = rc4([self.__diffie_key], message)

        request = {
            'type': 'client_message',
            'message': encrypted_message
        }

        self.send(request)

    def start_auth(self, login, password) -> tuple[str, str]:
        pass_check_result = self._send_login(login)

        if pass_check_result is None \
            or 'type' not in pass_check_result.keys() \
            or pass_check_result['type'] != 'pass_check':
            return ('failed', 'Пользователь с заданным логином не найден в базе данных')

        hashed_secret_word = pass_check_result['secret']

        session_status_result = self._check_hash(password, hashed_secret_word)

        if session_status_result is None \
            or 'type' not in session_status_result.keys() \
            or session_status_result['type'] != 'sess_keygen':
            return ('failed', 'Проверка не пройдена по причине внутренней ошибки сервера')

        if session_status_result['status'] != 'ok':
            return ('failed', 'Неверный пароль')

        g, p, A = session_status_result['g'], session_status_result['p'], session_status_result['A']

        auth_status_result = self._send_session_key_final(g, p)

        if auth_status_result is None \
            or 'type' not in auth_status_result.keys() \
            or auth_status_result['type'] != 'client_message':
            return ('failed', 'Не удалось создать защищенное соединение')

        if auth_status_result['status'] != 'ok':
            return ('failed', 'Не удалось создать ключ шифрования передачи сообщений')

        self.__diffie_key = binpow(A, self.__diffie_B, p)

        print(f'client: diffie key {self.__diffie_key}')

        return ('ok', 'Все отлично') # auth completed

    def start_messaging(self):
        while True:
            message = input()
            self.send_message(message)

    def start_listening(self):
        self.__message_thread = threading.Thread(target=Client._listen_to_messages, args=(self, self.__connection))
        self.__message_thread.start()

    def _listen_to_messages(self, ret_connection):
        while True:
            message = ret_connection.recv(1024)
            message = pickle.loads(message)
            decrypted_message = rc4([self.__diffie_key], message)
            print(decrypted_message)

class Client_app:
    def __init__(self) -> None:
        self.__client = Client()
        self.__client.connect()

    def start_login(self, login: str, password: str) -> str:
        status, reason = self.__client.start_auth(login, password)
        print(f'client: auth status {status}, reason {reason}')

        return status, reason

    def start_messaging(self) -> None:
        self.__client.start_listening()
        # self.__client.start_messaging()

    def send_message_to_server(self, message: str) -> None:
        self.__client.send_message(message)

app = Flask(__name__, template_folder='../front', static_folder='../front')
app.config.from_object({
    'CSRF_ENABLED': True,
    'CORS_HEADERS': 'Content-Type'
})

client = Client_app()

@app.route('/client_login', methods=['POST'])
def login():
    data_json = json.loads(request.get_data())
    login = data_json['login']
    password = data_json['password']

    status, reason = client.start_login(login, password)

    result = {
        'status': status,
        'reason': reason,
        'href': 'http://127.0.0.1:5500/front/messages.html'
    }

    if status == 'ok':
        client.start_messaging()
        pass

    print(result)

    return json.dumps(result)

@app.route('/send_message', methods=['POST'])
def send_message():
    data_json = json.loads(request.get_data())
    message = data_json['message']

    client.send_message_to_server(message)

    result = {'status': 'ok'}

    return json.dumps(result)

if __name__ == '__main__':
    app.run(host = "127.0.0.1", port = "5500")