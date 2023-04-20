import logging 
import os 
from datetime import datetime 


LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d")}._log'
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig( filename=LOG_FILE_PATH,
format='[%(asctime)s ] %(lineno)d %(name)s %(levelname)s : %(message)s',
level=logging.INFO  ,
datefmt='%Y-%m-%d %H:%M:%S'
)

# if __name__ == '__main__':
#     logging.info('logging is working just lunched') 