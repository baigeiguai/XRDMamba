import time
import logging

def print_info_before_epoch(epoch_idx,lr):
    logging.info("---------------[epoch_%d]start-----------------"%(epoch_idx+1))
    logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    logging.info("lr:%s"%lr)


def print_info_after_epoch(epoch_idx):
    logging.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    logging.info("---------------[epoch_%d]end-----------------"%(epoch_idx+1))