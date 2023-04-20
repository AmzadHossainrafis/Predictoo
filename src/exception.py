import sys
import logging
from src.logger import logging

def error_massage_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename 
    error_message = f'Error: {str(error)} in {file_name} at line {exc_tb.tb_lineno}'


    return error_message

class CustomException(Exception):
    def __init__(self, error_message , error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_massage_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return f'{self.error_message}' 
    

