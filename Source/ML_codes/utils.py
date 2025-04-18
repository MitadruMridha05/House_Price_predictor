import os
import sys
from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")


def read_sql_data():
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established")
        df=pd.read_sql_query("select * from house_dataset",mydb)
        print(df.head())        

        return df
    

    except Exception as ex:
        raise CustomException(ex,sys)