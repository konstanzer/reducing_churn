from src.prepare import telco_model_prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

import eli5


telco_df = telco_model_prep()


telco_df.info()


telco_df.head(1)


plt.figure(figsize=(9,7))
corr_mat = telco_df.corr()
np.fill_diagonal(corr_mat.values, 0) #we don't care about self-corralation
sns.heatmap(corr_mat);


#isolate the response variable before checking multicollinearity
churn = telco_df.pop("Churn_Yes")


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)


calc_vif(telco_df)


telco_df = telco_df.drop(['MonthlyCharges', 'StreamingTV_Yes', 'PhoneService_Yes',
                          'TotalCharges', 'InternetService_No'], axis=1)


calc_vif(telco_df)


plt.figure(figsize=(9,7))
corr_mat = telco_df.corr()
np.fill_diagonal(corr_mat.values, 0) #we don't care about self-corralation
sns.heatmap(corr_mat);


X_train, X_test, y_train, y_test = train_test_split(telco_df, churn, test_size=15/100,
                                                    random_state=36, stratify=churn)
#divide X_train and y_train into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/85,
                                                    random_state=36, stratify=y_train)


X_train.shape, X_test.shape, X_val.shape, y_test.shape, y_train.shape, y_val.shape


#ccp_alpha prunes weak links based on impurity
#a useless, 100% impure node splits evenly (50/50) whereas a 100% pure node classifies all samples to one class
#capping min_samples_leaf and max_depth prevents overfitting
dt = DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, ccp_alpha=.01, random_state=36).fit(X_train, y_train)
f"Decision tree accuracy: {dt.score(X_val, y_val)}"


plt.figure(figsize=(15,6))
plot_tree(dt, feature_names=X_val.columns, class_names=True);


rf = RandomForestClassifier(random_state=36).fit(X_train, y_train)
f"Tuned random forest accuracy: {rf.score(X_val, y_val)}"


param_grid = {'n_estimators': [100, 200, 500], #default 100 trees
               'max_depth': [5, 10, 20], #default is ALL THE WAY
               'min_samples_split': [2, 5, 10], #default 2
               'min_samples_leaf': [1, 2, 4]} #default 1
#randomly choose 27 out of 3*3*3*3 (81) possible models 
rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                               param_distributions=param_grid,
                               n_iter=27, cv=3, verbose=0,
                               random_state=36, n_jobs=-1) ##-1 means use all the processors
rf_random.fit(X_train, y_train)


params = rf_random.best_params_
params


rf = RandomForestClassifier(**params, random_state=36).fit(X_train, y_train)
f"Tuned random forest accuracy: {rf.score(X_val, y_val)}"


#ranked importance based on Gini index
eli5.show_weights(rf, feature_names=list(X_val.columns))


#k=8 was best paramater
knn = KNeighborsClassifier(n_neighbors=8, weights='uniform').fit(X_train, y_train)
f"kNN accuracy: {knn.score(X_val, y_val)}"


#linear models will not converge without scaling
lr = make_pipeline(StandardScaler(),LogisticRegression(random_state=36))
lr.fit(X_train, y_train)
f"Logistic regression accuracy: {lr.score(X_val, y_val)}"


eli5.show_weights(lr, feature_names=list(X_val.columns))


pd.DataFrame(X_train.columns)


small_train, small_val = X_train.iloc[:,[1,7]], X_val.iloc[:,[1,7]]
#skip scaling
lr2 = LogisticRegression(random_state=36)
lr2.fit(small, y_train)
f"Logistic regression accuracy: {lr2.score(small_val, y_val)}"


eli5.show_weights(lr2, feature_names=list(small_val.columns))


sgd = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-5, random_state=36)).fit(X_train, y_train)
f"Stochastic gradient descent accuracy: {sgd.score(X_val, y_val)}"


rr = make_pipeline(StandardScaler(),
                   RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=3)).fit(X_train, y_train)
f"Ridge regression accuracy: {rr.score(X_val, y_val)}"



