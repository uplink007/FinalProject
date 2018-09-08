import logging


class LoggerClass(object):
    def __init__(self, name="auto_de"):
        try:
            with open('../logs/auto_de.log', 'w'):
                pass
            self.name = name
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)
            self.fh = logging.FileHandler('../logs/auto_de.log')
            self.fh.setLevel(logging.DEBUG)
            self.ch = logging.StreamHandler()
            self.ch.setLevel(logging.CRITICAL)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.fh.setFormatter(formatter)
            self.ch.setFormatter(formatter)
            # add the handlers to the logger
            self.logger.addHandler(self.fh)
            self.logger.addHandler(self.ch)
        except Exception:
            self.logger.error("Logger initialization failed")
            raise
        self.logger.info("Logger initialized successfully")


if __name__ == "__main__":
    logger = LoggerClass(name="auto_de")
    pass
