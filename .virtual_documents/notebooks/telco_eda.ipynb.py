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








sns.set(style="darkgrid")

f=plt.subplots(figsize=(4,6))
ax=sns.countplot(data=df, x='internet_service', hue='churn', order=['fiber optic','DSL','none'])
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title("Churn by Internet Service");


df.Contract.value_counts()


f=plt.subplots(figsize=(4,6))
ax=sns.countplot(data=df, x='contract', hue='churn', order=['month-to-month','1-year','2-year'])
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title("Churn by Contract Type");


f=plt.subplots(figsize=(5,6))
ax=sns.countplot(data=df, x='payment_method', hue='churn', order=['e-check','check','bank transfer', 'credit card'])
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title("Churn by Payment Method");


f=plt.subplots(figsize=(5,6))
ax=sns.countplot(data=df, x='dependents', hue='churn')
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title("Churn by Dependents");


f=plt.subplots(figsize=(5,6))
ax=sns.countplot(data=df, x='online_security', hue='churn')
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title("Churn by Security");



