import socket
import pickle
from database_manager import DatabaseManager
from utils import Result
import os
import threading
from word_call_auth import generate_secret_word, unsign_data, sha1_combined
from SHA1 import sha1
from diffie_hellmann import generate_number, generate_g, generate_prime_numbers
from rsa import binpow
from rc4 import rc4
from flask import Flask, request
import json

class Server:
    def __init__(self, db: DatabaseManager, hostname='localhost', port=15002, max_connections=10) -> None:
        self.database = db
        self.__connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__hostname = hostname
        self.__port = port
        self.__max_conn = max_connections

        self.__ret_connection = None

        self.__diffie_key = None
        self.__diffie_A = None
        self.__diffie_p = None

        self.__secret = None
        self.__login = None

        self.message_buffer = []

    def start(self) -> None:
        self.__connection.bind((self.__hostname, self.__port))
        self.__connection.listen(self.__max_conn)
        connection = None
        threadPool = []
        
        print('server successfully started...')

        while True:
            connection, address = self.__connection.accept()
            thread = threading.Thread(target=Server._talk_to_client, args=(self, connection, address))
            self.__ret_connection = connection
            threadPool.append(thread)
            thread.start()

    def stop(self) -> None:
        self.__connection.close()

    def send(self, data, ret_connection) -> None:
        '''Send data via created socket connection'''
        buffer = pickle.dumps(data)
        ret_connection.send(buffer)
    
    def sendForResult(self, data, ret_connection, with_timeout: int or None = None) -> dict | None:
        '''Send data via socket connection and wait until response is coming'''
        self.send(data)
        # max_operation_count = with_timeout if with_timeout is not None else Client.__max_timeout
    
        ans_buffer = ret_connection.recv(1024)
        if len(ans_buffer) != 0:
            answer = pickle.loads(ans_buffer)
            return answer

        return None
    
    def check_if_user_exists(self, login: str) -> bool:
        result = self.database.get_data_from_table(f'select * from {self.database.users_table} where login="{login}";')
        if result is not None and len(result) != 0:
            return True

        return False
    
    def get_password_by_login(self, login: str) -> str:
        return self.database.get_data_from_table(f'select password from {self.database.users_table} where login="{login}";')[0, 0]

    def _talk_to_client(self, connection: socket, address: tuple) -> None:
        while True:
            if connection is None or not self._check_connection_alive(connection):
                return

            buffer = connection.recv(1024)
            # time.sleep(1)
            if len(buffer) > 0:
                try:
                    self._on_message(pickle.loads(buffer), connection, address)
                except:
                    print(bytes.decode(buffer, encoding='utf-8'))

    def send_message_to_client(self, message: str) -> None:
        encrypted_message = rc4([self.__diffie_key], message)
        self.__ret_connection.send(pickle.dumps(encrypted_message))

    def _check_connection_alive(self, conn: socket) -> bool:
        try:
            data = conn.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            if len(data) == 0:
                return False
        except BlockingIOError:
            return True
        except ConnectionResetError:
            return False
        except Exception as e:
            return True
        return True

    def _run_query(self, message: dict, ret_connection: socket) -> Result:
        if 'type' not in message.keys():
            return Result(error=ValueError('received message does not contain "type" field.\nAborting'))

        try:
            if message['type'] == 'auth':
                self._send_pass_check_begin(
                    login = message['login'],
                    ret_connection = ret_connection,
                )
            elif message['type'] == 'check_hash':
                self._send_sess_keygen_status(
                    H = message['H'],
                    hashed_password = message['data'],
                    encoded_hash = message['encoded_hash'],
                    e = message['public_key'],
                    N = message['N'],
                    secret = self.__secret,
                    ret_connection = ret_connection,
                )
            elif message['type'] == 'sess_keygen_final':
                self._send_message_start_status(
                    B = message['B'],
                    ret_connection=ret_connection,
                )
            elif message['type'] == 'client_message':
                decrypted_message = rc4([self.__diffie_key], message['message'])
                self.message_buffer.append(decrypted_message)
                print(f'server: recieved message: {decrypted_message}')
            else:
                raise Exception()
        except Exception as e:
            return Result(error=e)

    def _on_message(self, data: dict, ret_connection: socket, ret_address: tuple) -> None:
        # self.__connection.connect(ret_address)
        result = self._run_query(data, ret_connection)
        # ret_connection.send(pickle.dumps(result))

    def _send_pass_check_begin(self, login, ret_connection) -> None:
        if self.check_if_user_exists(login) != True:
            request = {
                'type': 'pass_check_failed',
            }

            self.send(request, ret_connection)
            return

        self.__login = login
        self.__secret = generate_secret_word()
        hashed_secret = sha1(self.__secret)

        print(f'server: secret word = {self.__secret}, hashed_secret = {hashed_secret}')

        request = {
            'type': 'pass_check',
            'secret': hashed_secret,
        }

        self.send(request, ret_connection)

    def _send_sess_keygen_status(
        self,
        H: str,
        hashed_password: str,
        encoded_hash: str,
        e: int,
        N: int,
        secret: str,
        ret_connection: socket
    ) -> None:

        print(f'server: hashed_password={hashed_password}, encoded_hash={encoded_hash}')

        server_hashed_password, decoded_hash = unsign_data(hashed_password, encoded_hash, e, N)

        print(f'server: hashed password = {server_hashed_password}, decoded hash = {decoded_hash}')

        if server_hashed_password != decoded_hash:
            request = {
                'type': 'sess_keygen',
                'status': 'failed',
            }

            self.send(request, ret_connection)
            return # someone changed our hashed password, aborting

        password = self.get_password_by_login(self.__login)

        server_hashed_password = sha1_combined(password, sha1(secret))

        print(f'server: local hashed password = {server_hashed_password}, client hashed password = {hashed_password}')

        if H != server_hashed_password:
            request = {
                'type': 'sess_keygen',
                'status': 'failed',
            }

            self.send(request, ret_connection)
            return # provided wrong password, aborting

        self.__diffie_A, self.__diffie_p = generate_number(128), generate_prime_numbers(512, 1)[0]
        g = generate_g(self.__diffie_p, 16)

        A = binpow(g, self.__diffie_A, self.__diffie_p)

        request = {
            'type': 'sess_keygen',
            'status': 'ok',
            'A': A,
            'g': g,
            'p': self.__diffie_p,
        }

        self.send(request, ret_connection)

    def _send_message_start_status(self, B: int, ret_connection: socket):
        self.__diffie_key = binpow(B, self.__diffie_A, self.__diffie_p)

        print(f'server: diffie key {self.__diffie_key}')

        request = {
            'type': 'client_message',
            'status': 'ok'
        }

        self.send(request, ret_connection)


class Server_app:

    def __init__(self) -> None:
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        config_initfile = os.path.join(current_folder_path, 'database_config.ini')

        manager = DatabaseManager()
        manager.add_config(config_initfile)
        manager.init_db_connection()

        self.__server = Server(manager)
    
    def start(self) -> None:
        self.__server_thread = threading.Thread(target=Server.start, args=(self.__server,))
        self.__server_thread.start()

    def start_login(self, login: str, password: str) -> str:
        is_user_exists = self.__server.check_if_user_exists(login)
        if not is_user_exists:
            return 'failed'

        db_password = self.__server.get_password_by_login(login)
        if password != db_password:
            return 'failed'

        return 'ok'
    
    def send_message(self, message: str) -> None:
        self.__server.send_message_to_client(message)

    def retrieve_messages(self) -> list:
        new_messages = self.__server.message_buffer.copy()
        self.__server.message_buffer = []
        return new_messages

app = Flask(__name__, template_folder='../front', static_folder='../front')
app.config.from_object({
    'CSRF_ENABLED': True,
    'CORS_HEADERS': 'Content-Type'
})

server = Server_app()
server.start()

@app.route('/client_login', methods=['POST'])
def login():
    data_json = json.loads(request.get_data())
    login = data_json['login']
    password = data_json['password']

    status = server.start_login(login, password)

    result = {
        'status': status,
        'href': 'http://127.0.0.1:5501/front/messages.html'
    }

    return json.dumps(result)

@app.route('/send_message', methods=['POST'])
def send_message():
    data_json = json.loads(request.get_data())
    message = data_json['message']

    server.send_message(message)

    result = {'status': 'ok'}

    return json.dumps(result)

@app.route('/check_messages', methods=['POST'])
def check_new_messages():
    new_messages = server.retrieve_messages()

    result = {'status': 'ok', 'messages': new_messages}

    return json.dumps(result)

if __name__ == '__main__':
    app.run(host = "127.0.0.1", port = "5501")