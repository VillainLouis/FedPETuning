""" pretty logging for FedETuning """

import sys
from loguru import logger
from utils.register import registry
import wandb

def formatter(record):
    # default format
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    if record["level"].name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger():
    logger.remove()

    logger.add(
        sys.stderr, format=formatter,
        colorize=True, enqueue=True
    )

    from datetime import datetime

    # 获取当前时间
    now = datetime.now()

    # 将时间格式化为字符串
    time_str = now.strftime("%Y-%m-%d--%H:%M")
    
    logger.add(
        "/data0/jliu/workspace/output/results/" + time_str + ".log",
        format=formatter,
        enqueue=True
    )

    registry.register("logger", logger)

    return logger
