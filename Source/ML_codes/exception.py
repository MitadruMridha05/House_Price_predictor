import sys
from Source.ML_codes.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Function to extract detailed error information including file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message  # Return the formatted error message

class CustomException(Exception):
    """
    Custom Exception class to handle exceptions with detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Initialize parent class with the error message
        self.error_message = error_message_detail(error_message, error_detail)  # Generate detailed error message

    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is converted to a string