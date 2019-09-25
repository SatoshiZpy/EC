# coding:utf-8
import logging
from logging import handlers


class Logger():
    level_relations = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING,
                       "error": logging.ERROR, "critical": logging.CRITICAL,}
    fmt_str = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s:%(message)s"
    def __init__(self, filename, level="info", when="D", backCount=2, fmt=fmt_str):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        self.th = handlers.TimedRotatingFileHandler(filename, when=when, backupCount=backCount, encoding="utf-8")
        self.th.setFormatter(format_str)

    def log_write(self, log_text, level="info"):
        self.logger.addHandler(self.th)
        if level == "debug" or level == "DEBUG":
            self.logger.debug(log_text)
        elif level == "info" or level == "INFO":
            self.logger.info(log_text)
        elif level == "warning" or level == "WARNING":
            self.logger.warning(log_text)
        elif level == "error" or level == "ERROR":
            self.logger.error(log_text)
        elif level == "critical" or level == "CRITICAL":
            self.logger.critical(log_text)
        else:
            raise ("日志级别错误")
        self.logger.removeHandler(self.th)  # 日志写入完成后移除handler


if __name__ == '__main__':
    log = Logger("text.log")
    log.log_write('5555')