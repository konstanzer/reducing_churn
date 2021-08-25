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

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

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


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)


calc_vif(telco_df)


telco_df = telco_df.drop(['MonthlyCharges', 'PhoneService_Yes', 'InternetService_No',
                          'TotalCharges'], axis=1)


calc_vif(telco_df)


plt.figure(figsize=(9,7))
corr_mat = telco_df.corr()
np.fill_diagonal(corr_mat.values, 0) #we don't care about self-corralation
sns.heatmap(corr_mat);


#isolate the response variable
churn = telco_df.pop("Churn_Yes")


#make a test set before scaling or oversampling the data
X_train, X_test, y_train, y_test = train_test_split(telco_df, churn, test_size=15/100,
                                                    random_state=36, stratify=churn)


churn.value_counts()


# define random oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
#X_train, y_train = oversample.fit_resample(X_train, y_train)
#or synthetic kNN data
smote = SMOTE(random_state=36)
X_train, y_train = smote.fit_resample(X_train, y_train)


y_train.value_counts() #oversampling the churn class


#divide X_train and y_train into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/85,
                                                    random_state=36, stratify=y_train)


X_train.shape, X_test.shape, X_val.shape, y_test.shape, y_train.shape, y_val.shape


#baseline: no churn
#tp / (tp + fp) is precision
#tp / (tp + fn) is recall
print(f"Accuracy: {1 - sum(y_test)/len(y_test)}")
print(f"Precision: {sum(y_test)/(sum(y_test)+0)}")
print(f"Recall: {sum(y_test)/(sum(y_test+0))}")
print(classification_report(y_test, np.zeros(len(y_test))))


#kNN benefited from the SMOTEing on dev set
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform').fit(X_train, y_train)
f"kNN accuracy: {knn.score(X_val, y_val)}"


dt = DecisionTreeClassifier(random_state=36).fit(X_train, y_train)
f"Overfit decision tree accuracy: {dt.score(X_val, y_val)}"


#ccp_alpha prunes weak links based on impurity, a useless, 100% impure node splits evenly (50/50) whereas a 100% pure node classifies all samples to one class
dt = DecisionTreeClassifier(ccp_alpha=.005, random_state=36).fit(X_train, y_train)
f"Decision tree accuracy: {dt.score(X_val, y_val)}"


plt.figure(figsize=(15,9))
plot_tree(dt, feature_names=X_val.columns, class_names=True);


rf = RandomForestClassifier(random_state=36).fit(X_train, y_train)
f"Overfit random forest accuracy: {rf.score(X_val, y_val)}"


#without oversampling, recall for churn was a dismal .26
rf = RandomForestClassifier(ccp_alpha=.005, random_state=36).fit(X_train, y_train)
print(classification_report(y_val, rf.predict(X_val)))
f"Random forest accuracy: {rf.score(X_val, y_val)}"


#ranked importance based on Gini index
eli5.show_weights(rf, feature_names=list(X_val.columns))


#wasn't converging at first because of collinearity
#again, oversampling gives big boost to churn recall, .51 to .85
#accuracy doesn't tell the story: baseline is 74% accurate with 
lr = make_pipeline(StandardScaler(),LogisticRegression(random_state=36))
#lr = LogisticRegression(random_state=36)
lr.fit(X_train, y_train)
print(classification_report(y_val, lr.predict(X_val)))
f"Logistic regression accuracy: {lr.score(X_val, y_val)}"


eli5.show_weights(lr, feature_names=list(X_val.columns))


sgd = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-5, random_state=36)).fit(X_train, y_train)
f"Stochastic gradient descent accuracy: {sgd.score(X_val, y_val)}"


rr = make_pipeline(StandardScaler(),
                   RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=3)).fit(X_train, y_train)
f"Ridge regression accuracy: {rr.score(X_val, y_val)}"






