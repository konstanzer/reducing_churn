import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


path = "/content/drive/MyDrive/TELCO/Kaggle_Telco.csv"
telco = pd.read_csv(path)


telco.info()


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=telco, x='InternetService', hue='Churn', palette='hot_r', stat='count', )
ax.set_xlabel("Plan")
ax.set_title("Churn by Internet Service");


telco.Contract.value_counts()


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=telco, x='Contract', hue='Churn', palette='hot_r')
ax.set_xlabel("Plan")
ax.set_title("Churn by Contract Type");


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=telco, x='PaymentMethod', hue='Churn', palette='hot_r')
ax.set_xlabel("Method")
ax.set_title("Churn by Payment Method");


f=plt.subplots(figsize=(15,7))
ax=sns.scatterplot(data=telco, y='MonthlyCharges', x='tenure', hue='Churn', palette='cubehelix_r')
ax.set_xlabel("Months with company")
ax.set_title("Monthly Charges Over Time");



