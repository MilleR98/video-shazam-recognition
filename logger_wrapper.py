import enum
import inspect
import logging.config
import os
import sys
from typing import List

import yaml


class LevelFilter(logging.Filter):
    def __init__(self, lowest_level):
        super().__init__()
        self.__lowest_level = lowest_level

    def filter(self, log_record):
        return log_record.levelno <= self.__lowest_level


class LogLevel(enum.Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


class Log:
    __cfg_loaded: bool = False

    @staticmethod
    def configure(path_to_config: str = 'logger_config.yml', root_dir_path: str = None):
        try:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_config), 'r') as f:
                configuration = yaml.safe_load(f.read())
                log_file_path = configuration['handlers']['file']['filename']

                if root_dir_path is not None:
                    if not os.path.exists(root_dir_path):
                        raise ValueError('Invalid root_dir path: ' + root_dir_path)
                    log_file_path = os.path.join(root_dir_path, log_file_path)
                    configuration['handlers']['file']['filename'] = log_file_path

                if not os.path.exists(os.path.dirname(log_file_path)):
                    os.makedirs(os.path.dirname(log_file_path))

                logging.config.dictConfig(configuration)

                __console_handler: logging.Handler = next(iter([h for h in logging.root.handlers if h.name == 'console']), None)
                if __console_handler is not None:
                    __console_handler.addFilter(LevelFilter(logging.INFO))

            Log.__cfg_loaded = True
        except Exception as e:
            logging.error('Logging configuration load failed ' + str(e), exc_info=True)

    def __init__(self, module_name=None, level: LogLevel = None) -> None:
        self.__warnings: List[str] = []

        if module_name is None:
            stk = inspect.stack()[1]
            mod = inspect.getmodule(stk[0])
            module_name = mod.__name__
        self.__logger_instance = logging.getLogger(module_name)
        self.setLevel(level)

    def __del__(self):
        if self.warnings:
            print('Attention: The logging class collected the following Warnings:\n%s' %
                  '\n'.join(['%u: %s' % (ind, warn_msg) for ind, warn_msg in enumerate(self.__warnings)]))

    def setLevel(self, log_level: LogLevel):
        if log_level is not None:
            self.__logger_instance.setLevel(log_level.value)

    @property
    def warnings(self) -> List[str]:
        return self.__warnings

    def info(self, msg: str):
        self.__logger_instance.info(msg=msg)

    def debug(self, msg: str):
        self.__logger_instance.debug(msg=msg)

    def warning(self, msg: str):
        self.__logger_instance.warning(msg=msg)
        self.__warnings.append(msg)

    def error(self, msg: str):
        self.__logger_instance.error(msg=msg)
        sys.exit()

    def critical(self, msg: str):
        self.__logger_instance.critical(msg=msg)
        sys.exit()
