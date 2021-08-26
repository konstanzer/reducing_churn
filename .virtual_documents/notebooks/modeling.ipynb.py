import warnings
warnings.filterwarnings('ignore')

from src.prepare import telco_model_prep
from src.helper_functions import plot_corr, calc_vif, output_preds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import eli5


df = telco_model_prep()
df.info()


df.head()


churn = df.pop("churn")


plt.figure(figsize=(9,7))
corr = df.corr()
sns.heatmap(corr, mask=np.triu(corr), annot=True);


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [round(variance_inflation_factor(X.values, i),1) for i in range(X.shape[1])]
    return(vif)


calc_vif(df.iloc[:,1:])


#df['streaming'] =  (df.streaming_movies + df.streaming_tv > 0).astype(int)
df = df.drop(['monthly_charges', 'total_charges'], axis=1)


calc_vif(df.iloc[:,1:])


plt.figure(figsize=(9,7))
corr = df.corr()
sns.heatmap(corr, mask=np.triu(corr), annot=True);


X_train, X_test, y_train, y_test = train_test_split(df, churn, test_size=.1,
                                                    random_state=36, stratify=churn)
X_train.pop("customer_id")
cid = X_test.pop("customer_id") #needed for output file
#make val before oversampling so test and val come from the same distribution
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9,
                                                    random_state=36, stratify=y_train)


#synthetic training data made with kNN algorithm
X_smote, y_smote = SMOTE(random_state=36).fit_resample(X_train, y_train)
y_smote.value_counts() #oversampling the churn class


X_train.shape, X_test.shape, X_val.shape, X_smote.shape, y_test.shape, y_train.shape, y_val.shape, y_smote.shape,


#baseline: no churn
#tp / (tp + fp) is precision & tp / (tp + fn) is recall
print(f"Accuracy: {np.round(1 - sum(y_test)/len(y_test), 2)}")
print(classification_report(y_test, np.zeros(len(y_test))))


scores_in, scores_out = [], []
k_values = range(1,31)

for k in k_values:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_smote, y_smote)
    scores_in.append(knn.score(X_train, y_train))
    scores_out.append(knn.score(X_val, y_val))                 


f = plt.subplots(figsize=(16,4))
plt.plot(k_values, scores_in)
plt.plot(k_values, scores_out)
plt.grid()
plt.xlim(1,30)
plt.legend(["train","dev"])
plt.title("kNN accuracy on training & validation sets for different values k");


k=20
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
knn.fit(X_train, y_train)
print(classification_report(y_val, knn.predict(X_val)))


#SMOTE data increases correct churn predictions and non-churn errors equally, I prefer this model over the one above but accuracy is too low
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
knn.fit(X_smote, y_smote)
print(classification_report(y_val, knn.predict(X_val)))


#ccp_alpha prunes weak links based on impurity, a useless, 100% impure node splits evenly (50/50) whereas a 100% pure node classifies all samples to one class
dt = DecisionTreeClassifier(ccp_alpha=.001, random_state=36)
dt.fit(X_train, y_train)
print(classification_report(y_val, dt.predict(X_val)))


dt = DecisionTreeClassifier(ccp_alpha=.001, random_state=36)
dt.fit(X_smote, y_smote)
print(classification_report(y_val, dt.predict(X_val)))


eli5.show_weights(dt, top=5, feature_names=list(X_train.columns))


plt.figure(figsize=(21,9))
plot_tree(dt, feature_names=X_val.columns, class_names=True);


rf = RandomForestClassifier(ccp_alpha=.005, random_state=36)
rf.fit(X_train, y_train)
print(classification_report(y_val, rf.predict(X_val)))


alphas = np.array(range(0, 480, 40))/10000

for a in alphas:
    rf = RandomForestClassifier(ccp_alpha=a, random_state=36)
    rf.fit(X_smote, y_smote)
    y_pred = rf.predict(X_val)
    print(f"{a}, recall: {round(recall_score(y_val, y_pred),2)},\
    precision: {round(precision_score(y_val, y_pred),2)},\
    accuracy: {round(accuracy_score(y_val, y_pred),2)}")    


alpha = .036

rf = RandomForestClassifier(ccp_alpha=alpha, random_state=36).fit(X_smote, y_smote)
y_pred = rf.predict(X_val)
print(classification_report(y_val, rf.predict(X_val)))


#ranked importance based on Gini index
eli5.show_weights(rf, feature_names=list(X_val.columns))


#output predictions
def output_preds(model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(f"{model}, \n    recall: {round(recall_score(y_test, y_pred),2)},\n\
    precision: {round(precision_score(y_test, y_pred),2)},\n\
    accuracy: {round(accuracy_score(y_test, y_pred),2)}") 
    print(f"\n customers: {len(y_test)}, predicted to churn: {sum(y_pred)}, did churn: {sum(y_test)}")
    output = pd.concat([cid.reset_index(drop=True),
                        pd.Series(np.round(y_proba[:,1], 3), name="probability"),
                        pd.Series(y_pred, name="label")], axis=1)
    output.to_csv(f'{model}_predictions.csv', index=False)


output_preds(rf)


#wasn't converging at first because of collinearity
#again, oversampling gives big boost to churn recall, .51 to .85
#accuracy doesn't tell the story: baseline is 74% accurate with 
lr = make_pipeline(StandardScaler(),LogisticRegression(C=.1, random_state=36))
lr.fit(X_train, y_train)
print(classification_report(y_val, lr.predict(X_val)))


lr = make_pipeline(StandardScaler(),LogisticRegressionCV(random_state=36))
lr.fit(X_smote, y_smote)
print(classification_report(y_val, lr.predict(X_val)))


eli5.show_weights(lr, feature_names=list(X_val.columns))


sgd = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=200, tol=1e-4, random_state=36)).fit(X_smote, y_smote)
print(classification_report(y_val, sgd.predict(X_val)))






