import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import math

path = "/home/andrea/Desktop/python/titanic/"
all_data = path + "train_and_test2.csv"

CSV_COLUMNS = [ "Passengerid","Age","Fare","Sex","sibsp","zero","zero","zero","zero","zero"
	,"zero","zero","Parch","zero","zero","zero","zero","zero","zero","zero","zero","Pclass"
	,"zero","zero","Embarked","zero","zero","2urvived"]

CSV_TRAIN = [ "Age","Fare","Sex", "Pclass","sibsp", "Parch"]
CSV_TARGET = ["2urvived"]
# Training examples
dataset = shuffle(pd.read_csv( all_data, names=CSV_COLUMNS, header=1, skipinitialspace=True))

data_len = len( dataset )

dataset_X = dataset[ CSV_TRAIN ]
dataset_Y = dataset[ CSV_TARGET ]


X_train = dataset_X[: math.ceil(data_len*2.0/3.0) ]
X_test = dataset_X[ - math.ceil(data_len/3.0) : ]


Y_train = dataset_Y[: math.ceil(data_len*2.0/3.0) ].values.ravel()
Y_test = dataset_Y[ - math.ceil(data_len/3.0) : ].values.ravel()


# Prams for classifier
n_estimators = 90
learning_rate = .03
max_depth = 4
subsample = 1
colsample_bytree = 1
gamma = 0
max_delta_step = 0
min_child_weight = 1

# Build and fit model
model = XGBClassifier(n_estimators = n_estimators,
                      max_depth = max_depth,
                      learning_rate = learning_rate,
                      subsample = subsample,
                      colsample_bytree = colsample_bytree,
                      gamma = gamma,
                      max_delta_step = max_delta_step,
                      min_child_weight = min_child_weight
                      )

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

ko = 0
ok = 0


for i in range( 0, len(Y_test) ):
	if(y_pred[i] != Y_test[i]):
		#print( "KO    )      " + str(y_pred[i]) +  "  - not equal - " + str(Y_test[i]) )
		ko += 1
	else:
		#print( "OOOOOOK    )      " + str(y_pred[i]) +  "  - equal - " + str(Y_test[i]) )
		ok +=1


#print("\n\n ko is : " + str(ko) + ";       whilst ok is: " + str( ok )  )
percentage = (ok/(ok+ko))*100.0
print("\n correctness percentage is: " + str( percentage ) )
