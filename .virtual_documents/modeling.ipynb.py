from src.prepare import telco_model_prep
import pandas as pd
import numpy as np

#Estimator that has built-in cross-validation capabilities to automatically select the best hyper-parameters
#Roughly equivalent to GridSearchCV(Estimator(), ...).
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


telco_df = telco_model_prep()


telco_df.info()


telco_df.head()


churn = np.array(telco_df.pop("Churn_Yes"))


#algorithm will not converge without this step
telco_df = scale(telco_df) #indpendently standardizes each feature (mu=0, var=1)


X_train, X_test, y_train, y_test = train_test_split(telco_df, churn, test_size=0.2, random_state=36, stratify=churn)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


X_train[0], y_train[0]


clf = LogisticRegressionCV(cv=5, scoring='accuracy', random_state=36).fit(X_train, y_train)


clf.score(X_test, y_test)



