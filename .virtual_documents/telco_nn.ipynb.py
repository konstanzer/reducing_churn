import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


path = "/content/drive/MyDrive/TELCO/telco_numeric.csv"
telco = pd.read_csv(path, index_col=0)


telco.head()


1 - sum(y_test)/len(y_test) #baseline accuracy


#multicollinear variables determined with inflation factors
telco = telco.drop(['MonthlyCharges', 'PhoneService_Yes',
                    'TotalCharges', 'InternetService_No'], axis=1)


features = telco.copy()
labels = features.pop('Churn_Yes')
#features = StandardScaler().fit_transform(features) #this can be done with the model itself
features = np.array(features)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.05, stratify=labels, random_state=42)


smote = SMOTE(random_state=36) #this made model rather worse
#X_train, y_train = smote.fit_resample(X_train, y_train)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


n_features = X_train.shape[1]
n_classes = 2
# define the keras model
#https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/
model = Sequential()
model.add(Normalization())
model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(n_classes, activation='softmax')) #this has to have 1 unit for binary classifier


# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=30, batch_size=256, verbose=2)


# evaluate on test set
yhat = model.predict(X_test)
yhat = np.argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(y_test, yhat)
print('Accuracy: get_ipython().run_line_magic(".3f'", " % acc)")


print(classification_report(y_test, yhat))


print(confusion_matrix(y_test, yhat))


model.predict(X_test)


#from Anderson and Guvenc model
ann = Sequential()
ann.add(Dense(units = 200,activation="relu", kernel_regularizer=l2(0.001)))
ann.add(Dropout(0.2))
ann.add(Dense(units = 100, activation="relu",kernel_regularizer=l2(0.001)))
ann.add(Dropout(0.2))
ann.add(Dense(units = 50, activation="relu",kernel_regularizer=l2(0.001)))
ann.add(Dense(1, activation="sigmoid"))

ann.compile(optimizer = "adam", loss="binary_crossentropy",metrics=["accuracy"])

callback = EarlyStopping(monitor="val_loss", patience=2)
history = ann.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),
                  batch_size=16, epochs=5, callbacks=[callback])


pd.DataFrame(ann.history.history).plot(figsize=(15,10))


predictions = ann.predict_classes(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))



