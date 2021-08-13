import pandas as pd

#Link to download dataset: https://www.kaggle.com/blastchar/telco-customer-churn
def get_telco_data():
    path = "~/Documents/Github/school/Telco/data/" #path to download location
    return pd.read_csv(path + "Kaggle_Telco.csv")

if __name__ == '__main__':
    
    df = get_telco_data()
    print(df.head())
    