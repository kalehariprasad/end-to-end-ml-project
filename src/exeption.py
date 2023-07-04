import sys
from logger.py import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error occurred in script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        error_message = error_message_detail(error, error_detail=error_detail)
        super().__init__(error_message)
        self.error_message = error_message

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.exception("An error occurred:")
        raise CustomException(e, error_detail=sys.exc_info())


