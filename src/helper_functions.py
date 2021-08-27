import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def plot_corr(X):
    #A lower-triangle correlation heatmap
    plt.figure(figsize=(11,7))
    corr = X.corr()
    return sns.heatmap(corr,mask=np.triu(corr),annot=True)

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [round(variance_inflation_factor(X.values, i),1) for i in range(X.shape[1])]
    return(vif)

def over_split(X, y, test_size):
    #Oversamples training data using SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=36, stratify=y)
    X_train.pop("customer_id")
    #make val before oversampling so test and val come from the same distribution
    test_size2 = test_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size2,
                                                        random_state=36, stratify=y_train)
    #synthetic training data made with kNN algorithm
    X_over, y_over = SMOTE(random_state=36).fit_resample(X_train, y_train)
    
    print("X_train, X_test, y_train, y_test, X_over, X_val, y_over, y_val")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape,
          X_over.shape, X_val.shape, y_over.shape, y_val.shape)
    return X_train, X_test, y_train, y_test, X_over, X_val, y_over, y_val


def chi2_significance(df, y, col_start=1, col_end=None):
    insignificant_variables = []
    
    for col in df.columns[col_start:col_end]:
        observed = pd.crosstab(y, df[col])
        chi2, p, degf, expected = chi2_contingency(observed)
        #Bonferroni correction = to no. of comparisons
        bonf, alpha = observed.shape[0]*observed.shape[1], .05
        if bonf*p > alpha:
            insignificant_variables.append(col)
            print('Observed')
            print(observed)
            print('---\nExpected')
            print(pd.DataFrame(np.round(expected), index=['No', 'Yes']))
            print(f'---\nBonferroni correction = {bonf}')
            print(f'chi^2 = {chi2:.3f}')
            print(f'p     = {np.min([bonf*p, .9999]):.4f}\n\n')
            
    return insignificant_variables


def bar_plot():
    pass


def output_preds(model, X_test, y_test, cid):
    #Print test results and save predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(f"{model}, \n    recall: {round(recall_score(y_test, y_pred),2)},\n\
    precision: {round(precision_score(y_test, y_pred),2)},\n\
    f1: {round(f1_score(y_test, y_pred),2)},\n\
    accuracy: {round(accuracy_score(y_test, y_pred),2)}") 
    
    output = pd.concat([cid.reset_index(drop=True),
                        pd.Series(np.round(y_proba[:,1], 3), name="probability"),
                        pd.Series(y_pred, name="label")], axis=1)
    output.to_csv(f'{model}_predictions.csv', index=False)