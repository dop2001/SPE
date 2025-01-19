from torch.utils.tensorboard import SummaryWriter
import logging


class Loger:
    def __init__(self, loger_config):
        self.loger_config = loger_config
        # log to file
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(self.loger_config["log_path"]), logging.StreamHandler()])
        self.loger = logging.getLogger(name='good_luck')

    def getLoger(self):
        self.loger.info('Logging to file...')
        return self.loger


class TensorboardWriter:
    def __init__(self, summaryWriter_config):
        self.loger_config = summaryWriter_config
        # tensorboard writer
        self.summaryWriter = SummaryWriter(self.loger_config["summary_path"])

    def getSummaryWriter(self):
        return self.summaryWriter

