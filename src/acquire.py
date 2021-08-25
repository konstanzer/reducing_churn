try: from src.env import host, username, password
except: from env import host, username, password
import pandas as pd

#Link to download dataset: https://www.kaggle.com/blastchar/telco-customer-churn
def get_telco_data_kaggle():
    path = "~/Documents/Github/school/Telco/data/" #path to download location
    return pd.read_csv(path + "Kaggle_Telco.csv")

def get_db_url(username, host, password, db):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_telco_data():
    url = get_db_url(username, host, password, 'telco_churn')
    query = """SELECT * FROM customers;"""
    return pd.read_sql(query, url)


if __name__ == '__main__':
    
    df = get_telco_data()
    print(df.head())
    