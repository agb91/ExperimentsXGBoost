from  __future__ import division
import math
from numpy import loadtxt
import numpy as np
from gene import Gene
from sklearn.preprocessing import LabelEncoder  
import pandas as pd
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np


class TitanicBoost:

	def __init__( self  ):
		self.col_by_tree = None
		self.subsample = None
		self.min_child_weight = None
		self.max_depth = None
		self.n_estimators = None
		self.learning_rate = None
		self.X_train = None
		self.Y_train = None
		self.X_test = None
		self.Y_test = None

	def setGene( self, gene ):
		self.col_by_tree = gene.col_by_tree
		self.subsample = gene.subsample
		self.min_child_weight = gene.min_child_weight
		self.max_depth = gene.max_depth
		self.n_estimators = gene.n_estimators
		self.learning_rate = gene.learning_rate
		
	def rounder( self, x , ts ):
		if( x > ts ):
			return 1
		else:
			return 0	

	def manageDatasets(self):

		le = LabelEncoder()
		
		path = "/home/andrea/Desktop/python/titanic/"
		train_data = path + "train.csv"
		test_data = path + "test.csv"

		CSV_COLUMNS = [ "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
		CSV_COLUMNS_TEST = [ "PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]



		CSV_TRAIN = [ "Age","Fare","Sex", "Pclass","SibSp", "Parch", "Cabin"]
		CSV_TARGET = ["Survived"]
		
		train_valid = shuffle(pd.read_csv( train_data, names=CSV_COLUMNS, header=1, skipinitialspace=True)).fillna(0)
		train_valid['Sex'] = pd.Categorical.from_array(train_valid.Sex).codes
		train_valid['Cabin'] = train_valid['Cabin'].apply(lambda x: str(x)[:1])
		train_valid['Cabin'] = pd.Categorical.from_array(train_valid.Cabin).codes
		
		data_len = len(train_valid)
		dataset_train = train_valid[: math.ceil(data_len*2.0/3.0) ]
		dataset_valid = train_valid[ - math.ceil(data_len/3.0) : ]
		dataset_test = shuffle(pd.read_csv( test_data, names=CSV_COLUMNS_TEST, header=1, skipinitialspace=True)).fillna(0)
		dataset_test['Sex'] = pd.Categorical.from_array(dataset_test.Sex).codes
		dataset_test['Cabin'] = dataset_test['Cabin'].apply(lambda x: str(x)[:1])
		dataset_test['Cabin'] = pd.Categorical.from_array(dataset_test.Cabin).codes
		
		
		self.X_train = dataset_train[ CSV_TRAIN ]
		self.Y_train = dataset_train[ CSV_TARGET ].values.ravel()

		self.X_valid = dataset_valid[ CSV_TRAIN ]
		self.Y_valid = dataset_valid[ CSV_TARGET ].values.ravel()

		self.X_test = dataset_test[ CSV_TRAIN ]
		


	def run( self, type ):

		
		#print( Y_test )

		#print (X_train)
		if( type==0 ):
			gbm = xgb.XGBRegressor( colsample_bytree = self.col_by_tree, subsample = self.subsample, 
				min_child_weight = self.min_child_weight , max_depth=self.max_depth, 
				n_estimators=self.n_estimators,
				learning_rate=self.learning_rate).fit(self.X_train, self.Y_train)
		
		if( type==1 ):
			gbm = xgb.XGBClassifier( colsample_bytree = self.col_by_tree, subsample = self.subsample, 
				min_child_weight = self.min_child_weight , max_depth=self.max_depth, 
				n_estimators=self.n_estimators,
				learning_rate=self.learning_rate).fit(self.X_train, self.Y_train)
		


		y_pred = gbm.predict(self.X_valid)

		y_pred = list( map( lambda p: self.rounder(p, 0.5) , y_pred ) )

		#print( y_pred )


		ko = 0
		ok = 0


		for i in range( 0, len(self.Y_valid) ):
			if(y_pred[i] != self.Y_valid[i]):
				#print( "KO    )      " + str(y_pred[i]) +  "  - not equal - " + str(Y_test[i]) )
				ko += 1
			else:
				#print( "OOOOOOK    )      " + str(y_pred[i]) +  "  - equal - " + str(Y_test[i]) )
				ok +=1


		#print("\n\n ko is : " + str(ko) + ";       whilst ok is: " + str( ok )  )
		percentage = (ok/(ok+ko))*100.0
		
		return percentage




