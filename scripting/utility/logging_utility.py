import logging
import os
import sys
import traceback
from logging import FileHandler
from typing import AnyStr

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


class SimpleLevelFilter:
    """
    Simple logging filter
    """

    def __init__(
            self,
            level
    ):
        self._level = level

    def filter(
            self,
            log_record
    ):
        """
        Filters log message according to filter level

        Args:
            log_record: message to log

        Returns:
            True if message level is less than or equal to filter level
        """

        return log_record.levelno <= self._level


class Logger:
    _instance = None
    _log_path = None

    _formatter = None
    _file_handler = None
    _stdout_handler = None

    @classmethod
    def _handle_exception(
            cls,
            exctype,
            value,
            tb
    ):
        if cls._instance is not None:
            cls._instance.info(f"Type: {exctype}{os.linesep}"
                               f"Value: {value}{os.linesep}"
                               f"Traceback: {''.join(traceback.format_exception(exctype, value, tb))}{os.linesep}")

    @classmethod
    def _set_file_handler(
            cls,
            log_path: AnyStr
    ):
        file_handler = FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(cls._formatter)

        return file_handler

    @classmethod
    def _build_logger(
            cls,
            name: str
    ):
        """
        Returns a logger instance that handles info, debug, warning and error messages.

        Args:
            name: logger name

        Returns:
            logger instance
        """

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.ERROR)
        cls._formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(cls._formatter)

        logger.addHandler(stream_handler)

        cls._file_handler = cls._set_file_handler(log_path=cls._log_path)
        logger.addHandler(cls._file_handler)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(cls._formatter)
        stdout_handler.addFilter(SimpleLevelFilter(logging.WARNING))
        cls._stdout_handler = stdout_handler
        logger.addHandler(stdout_handler)

        sys.excepthook = cls._handle_exception

        return logger

    @classmethod
    def set_log_path(
            cls,
            name: str,
            log_path: AnyStr = None
    ):
        log_path = os.path.join(log_path, 'daily_log.log')

        # Skip if re-declaration
        if cls._log_path is not None and cls._log_path == log_path:
            return

        cls._log_path = log_path
        cls.get_logger(name=name)

        if not os.path.isdir(log_path):
            cls._instance.info(f'Provided log_path={log_path} is not a directory. Maintaining old directory...')
        else:
            cls._instance.info(f'New file handler has been defined. Logging will be redirected to: {log_path}')

            # Refresh handler
            cls._instance.removeHandler(cls._file_handler)
            cls._file_handler = cls._set_file_handler(log_path=cls._log_path)
            cls._instance.addHandler(cls._file_handler)

    @classmethod
    def get_logger(
            cls,
            name: str
    ):
        if cls._instance is None:
            cls._instance = cls._build_logger(name)
            cls._instance.info(f'Retrieving new logger: {cls._log_path}')
        return cls._instance
