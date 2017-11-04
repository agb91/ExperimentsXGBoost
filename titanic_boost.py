from  __future__ import division
import math
from numpy import loadtxt
import numpy as np
from gene import Gene

import pandas as pd
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np


class TitanicBoost:

	def __init__( self , gene ):
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


	def run( self ):

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

		#print( Y_test )

		#print (X_train)

		gbm = xgb.XGBRegressor( colsample_bytree = self.col_by_tree, subsample = self.subsample, 
			min_child_weight = self.min_child_weight , max_depth=self.max_depth, 
			n_estimators=self.n_estimators,
			learning_rate=self.learning_rate).fit(X_train, Y_train)
		
		y_pred = gbm.predict(X_test)

		y_pred = list( map( lambda p: self.rounder(p, 0.5) , y_pred ) )

		#print( y_pred )


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
		
		return percentage




