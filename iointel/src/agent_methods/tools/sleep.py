import time

import logging

logger = logging.getLogger(__name__)

class SleepTools:
    def __init__(self):
        pass

    def sleep(self, seconds: int) -> str:
        """Use this function to sleep for a given number of seconds."""
        logger.info(f"Sleeping for {seconds} seconds")
        time.sleep(seconds)
        logger.info(f"Awake after {seconds} seconds")
        return f"Slept for {seconds} seconds"
