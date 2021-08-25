import warnings
warnings.filterwarnings('ignore')

from src.prepare import telco_eda_prep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = telco_eda_prep()
df.info()


df.head()


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=df, x='InternetService', hue='Churn', palette='hot_r', stat='count', )
ax.set_xlabel("Plan")
ax.set_title("Churn by Internet Service");


df.Contract.value_counts()


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=df, x='Contract', hue='Churn', palette='hot_r')
ax.set_xlabel("Plan")
ax.set_title("Churn by Contract Type");


f=plt.subplots(figsize=(9,6))
ax=sns.histplot(data=df, x='PaymentMethod', hue='Churn', palette='hot_r')
ax.set_xlabel("Method")
ax.set_title("Churn by Payment Method");


f=plt.subplots(figsize=(15,7))
ax=sns.scatterplot(data=df, y='MonthlyCharges', x='tenure', hue='Churn', palette='cubehelix_r')
ax.set_xlabel("Months with company")
ax.set_title("Monthly Charges Over Time");



