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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

class TitanicBoostClassifier:

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
		self.test_results = None
		self.X_output = None
		self.gbm = None

	def set_gene_to_model( self, gene ):
		self.col_by_tree = gene.col_by_tree
		self.subsample = gene.subsample
		self.min_child_weight = gene.min_child_weight
		self.max_depth = gene.max_depth
		self.n_estimators = gene.n_estimators
		self.learning_rate = gene.learning_rate
		self.gbm = xgb.XGBClassifier( colsample_bytree = self.col_by_tree, subsample = self.subsample, 
				min_child_weight = self.min_child_weight , max_depth=self.max_depth, 
				n_estimators=self.n_estimators, objective="binary:logistic" ,
				learning_rate=self.learning_rate)

		
	def rounder( self, x , ts ):
		if( x > ts ):
			return 1
		else:
			return 0

				

	def setDatasets( self, X , Y , X_test , X_output ):
		self.X = X
		self.Y = Y
		self.X_test = X_test
		self.X_output = X_output
		

	def predict( self ):
		
		self.gbm.fit(self.X, self.Y)
		y_pred_test = self.gbm.predict(self.X_test)

		ids = self.X_test['PassengerId']
		d = {'PassengerId': ids , 'Survived': y_pred_test}
		self.test_results = pd.DataFrame(data=d)

	def run( self ):
		kfold = StratifiedKFold(n_splits=5, random_state=7)
		results = cross_val_score( self.gbm, self.X, self.Y, cv=kfold, scoring='accuracy')

		return results.mean()




