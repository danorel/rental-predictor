from os import environ


class Config(object):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    SESSION_COOKIE_SECURE = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    HOST = '127.0.0.1'
    PORT = 8080
    BASE_URL = '/api/v1'
    SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
    PATH_TO_CSV = environ.get('PATH_TO_CSV')
    CSV_FILENAME = environ.get('CSV_FILENAME')
    PATH_TO_CONFIGURATIONS = environ.get('PATH_TO_CONFIGURATIONS')


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    TESTING = True
    SESSION_COOKIE_SECURE = False