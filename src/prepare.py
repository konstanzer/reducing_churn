from src.acquire import get_telco_data
import pandas as pd
import numpy as np


def telco_model_prep():
    
    telco = get_telco_data()
    
    telco = telco.drop(['customerID'], axis=1)
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'TotalCharges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    #conver all 15 object types into dummies
    telco = pd.get_dummies(telco, drop_first=True)
    
    return telco


if __name__ == '__main__':
    
    df = telco_model_prep()
    print(df.head())
    
    