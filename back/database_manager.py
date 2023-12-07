import configparser
import sqlalchemy
import numpy as np
import os

class Config:
    def __init__(self) -> None:
        self.username = None
        self.password = None
        self.SQLALCHEMY_DATABASE_URI = None
        self.CSRF_ENABLED = None

    def from_config_object(self, config: configparser.ConfigParser) -> None:
        try:
            self.username = config.get('security_config', 'username')
            self.password = config.get('security_config', 'password')
            self.SQLALCHEMY_DATABASE_URI = config.get('security_config', 'SQLALCHEMY_DATABASE_URI')
            self.CSRF_ENABLED = config.get('security_config', 'CSRF_ENABLED')
        except:
            print('Exception occured when parsing config from dictionary')

class DatabaseManager:
    def __init__(self) -> None:
        pass

    def add_config(self, config_filepath: str) -> None:
        parser = configparser.ConfigParser()
        try:
            parser.read(config_filepath)
        except:
            print("Configuration cannot be read. Invalid path provided")
            return
        
        self._config = Config()
        self._config.from_config_object(parser)

    def init_db_connection(self) -> None:
        if self._config is None:
            print("Exception occured: configuration is not set. Database connection is not established")
        else:
            self.__database = sqlalchemy.create_engine(self._config.SQLALCHEMY_DATABASE_URI)

    def get_data_from_table(self, query) -> np.ndarray:
        db = self.__require_not_none(self.__database)
        connection = db.connect()
        result = np.array(connection.execute(query).fetchall())
        connection.close()
        return result

    def execute_query(self, query) -> None:
        db = self.__require_not_none(self.__database)
        connection = db.connect()
        connection.execute(query)
        connection.close()

    def __require_not_none(self, obj):
        if obj is None:
            raise ValueError('Error occured: database is not initialized')
        return obj
    
if __name__ == '__main__':
    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    config_initfile = os.path.join(current_folder_path, 'database_config.ini')

    manager = DatabaseManager()
    manager.add_config(config_initfile)
    manager.init_db_connection()