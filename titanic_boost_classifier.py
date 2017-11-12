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

	def extract_name(self , name):
		part = name.split(',')[1].split('.')[0].strip()
		#print("|" + str(part) + "| --- " + "Mr" + str( part == "Mr" ))
		if( str(part) == "Mr" or str(part) == "Miss" or str(part) == "Mrs" or 
			str(part) == "Mme" or str(part) == "Mlle" or str(part) == "Ms" or
			str(part) == "Master" ):
			return 0
		else:
			return part


	def extract_ticket( self, ticket ):
		ticket = str(ticket)[:1].strip()
		if( ticket.isdigit() ):
			return 0
		else:
			return ticket


	def extract_age( self, age ):
		return age	

	def get_cabin( self,x ):
		val = str(x)[:1]
		return val	

	def manage_sex(self, train_valid):
		train_valid['Sex'] = pd.Categorical.from_array(train_valid.Sex).codes
		train_valid = pd.get_dummies(train_valid, columns = ["Sex"])
		return train_valid		
	
	def manage_cabin(self, train_valid):
		train_valid['Cabin'] = train_valid['Cabin'].apply(lambda x: self.get_cabin(x) )
		train_valid = pd.get_dummies(train_valid, columns = ["Cabin"])
		return train_valid		
		
	def manage_name(self, train_valid):
		train_valid['Name'] = train_valid['Name'].apply( lambda x: self.extract_name( x ) )
		train_valid = pd.get_dummies(train_valid, columns = ["Name"])
		return train_valid		

	def manage_ticket( self, train_valid ):
		train_valid['Ticket'] = train_valid['Ticket'].apply( lambda x: self.extract_ticket( x ) )
		train_valid = pd.get_dummies(train_valid, columns = ["Ticket"])
		return train_valid		
		
	def manage_age( self, train_valid ):
		train_valid['Age'] = train_valid['Age'].apply( lambda x: self.extract_age( x ) )
		return train_valid

	def manage_embarked( self, train_valid ):
		train_valid = pd.get_dummies(train_valid, columns = ["Embarked"])
		return train_valid
				

	def manageDatasets(self):

		le = LabelEncoder()
		
		path = "/home/andrea/Desktop/python/titanic/"
		train_data = path + "train.csv"
		test_data = path + "test.csv"

		CSV_COLUMNS = [ "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
		CSV_COLUMNS_TEST = [ "PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

		CSV_OUTPUT = ["PassengerId"]

		CSV_TARGET = ["Survived"]
		
		train_valid = pd.read_csv( train_data, names=CSV_COLUMNS, header=0, skipinitialspace=True)
		dataset_test = pd.read_csv( test_data, names=CSV_COLUMNS_TEST, header=0, skipinitialspace=True)
		
		global_dataset = pd.concat( [train_valid, dataset_test] )


		global_dataset = self.manage_sex(global_dataset)
		global_dataset = self.manage_cabin(global_dataset)
		global_dataset = self.manage_name(global_dataset)
		global_dataset = self.manage_ticket(global_dataset)
		global_dataset = self.manage_age(global_dataset)
		global_dataset = self.manage_embarked( global_dataset )


	
		self.Y = train_valid[ CSV_TARGET ].values.ravel()

		self.X = global_dataset.head( 891 )
		self.X.drop("Survived", axis=1, inplace=True)
		
		self.X_test = global_dataset.tail( 418 )
		self.X_test.drop("Survived", axis=1, inplace=True)
		self.X_output = self.X_test[ CSV_OUTPUT ]
		

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




