from datetime import datetime

from iointel.src.utilities.decorators import register_tool


@register_tool
def what_time_is_it():
    """
    Use this function to get current datetime
    """
    return datetime.now()
