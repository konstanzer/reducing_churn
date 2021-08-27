try: from src.acquire import get_telco_data
except: from acquire import get_telco_data
import pandas as pd
import numpy as np

def telco_model_prep():
    
    telco = get_telco_data()
    cid = telco.pop("customer_id")
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'total_charges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==3] = 0 #I'll drop this later
    telco.payment_type_id.loc[telco.payment_type_id==4] = 0
    telco.contract_type_id.loc[telco.contract_type_id==3] = 0
    telco.payment_type_id = telco.payment_type_id.astype(object)
    telco.contract_type_id = telco.contract_type_id.astype(object)
    telco.internet_service_type_id = telco.internet_service_type_id.astype(object)
    
    #convert all 15 object types into dummies
    telco = pd.get_dummies(telco, drop_first=True)
    telco = telco.rename(columns={"churn_Yes": "churn", "gender_Male": "male",
                                  'internet_service_type_id_1': "dsl",
                                  'internet_service_type_id_2': "fiber_optic",
                                  'contract_type_id_1':"one_month",
                                  'contract_type_id_2':"one_year",
                                  'payment_type_id_1': "e_check",
                                  'payment_type_id_2': "check",
                                  'payment_type_id_3': "bank_transfer",
                                  'partner_Yes':"partner", 'dependents_Yes':"dependents",
                                  'phone_service_Yes':"phone_service",
                                  'multiple_lines_Yes':"multiple_lines",
                                  'online_security_Yes':"online_security",
                                  'online_backup_Yes':"online_backup",
                                  'device_protection_Yes':"device_protection",
                                  'tech_support_Yes':"tech_support",
                                  'streaming_tv_Yes':"streaming_tv",
                                  'streaming_movies_Yes':"streaming_movies",
                                  'paperless_billing_Yes':"paperless"})
    
    return pd.concat([cid, telco], axis=1)


def telco_eda_prep():
    
    telco = get_telco_data()
    
    telco = telco.drop(['customer_id', 'senior_citizen'], axis=1)
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'total_charges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==1] = "DSL"
    telco.payment_type_id.loc[telco.payment_type_id==1] = "e-check"
    telco.contract_type_id.loc[telco.contract_type_id==1] = "month-to-month"
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==2] = "fiber optic"
    telco.payment_type_id.loc[telco.payment_type_id==2] = "check"
    telco.contract_type_id.loc[telco.contract_type_id==2] = "1-year"
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==3] = "none"
    telco.payment_type_id.loc[telco.payment_type_id==3] = "bank transfer"
    telco.contract_type_id.loc[telco.contract_type_id==3] = "2-year"
    
    telco.payment_type_id.loc[telco.payment_type_id==4] = "credit card"
    
    telco['monthly_charges_bins'] = pd.qcut(telco.monthly_charges, 10)
    
    telco = telco.rename(columns={"payment_type_id": "payment_method",
                                  "contract_type_id": "contract",
                                  'internet_service_type_id': "internet_service"})
    
    return telco


if __name__ == '__main__':
    
    df = telco_model_prep()
    print(df.head())
    print(df.columns)
    df.to_csv("telco_num.csv")
    
    