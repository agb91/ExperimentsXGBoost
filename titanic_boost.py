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
				n_estimators=self.n_estimators,
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
			str(part) == "Master" ):
			return 0
		else:
			return 1


	def extract_ticket( self, ticket ):
		ticket = str(ticket)[:1].strip()
		if( ticket.isdigit() ):
			return 0
		else:
			return 1


	def extract_age( self, age ):
		#child
		if( age < 13 ):
			return 0
		#elder	
		if( age > 60):
			return 3		
		#young	
		if( age >= 13 and age < 40 ):
			return 1
		#adult	
		return 2	

	def manage_sex(self, train_valid):
		train_valid['Sex'] = pd.Categorical.from_array(train_valid.Sex).codes
		dummies_sex = pd.get_dummies( train_valid['Sex']  )
		train_valid['Sex0'] = dummies_sex[dummies_sex.columns[0]] 
		train_valid['Sex1'] = dummies_sex[dummies_sex.columns[1]] 
		train_valid.drop('Sex', axis=1, inplace=True)
		return train_valid		
	
	def get_cabin( self,x ):
		val = str(x)[:1]
		if( val == 'n' ):
			return 0
		else:
			return 1	

	def manage_cabin(self, train_valid):
		train_valid['Cabin'] = train_valid['Cabin'].apply(lambda x: self.get_cabin(x) )
		dummies_cabin = pd.get_dummies( train_valid['Cabin']  )
		train_valid['Cabin0'] = dummies_cabin[dummies_cabin.columns[0]] 
		train_valid['Cabin1'] = dummies_cabin[dummies_cabin.columns[1]] 
		train_valid.drop('Cabin', axis=1, inplace=True)
		return train_valid		
		
	def manage_name(self, train_valid):
		train_valid['Name'] = train_valid['Name'].apply( lambda x: self.extract_name( x ) )
		dummies_name = pd.get_dummies( train_valid['Name']  )
		train_valid['Name0'] = dummies_name[dummies_name.columns[0]] 
		train_valid['Name1'] = dummies_name[dummies_name.columns[1]] 
		train_valid.drop('Name', axis=1, inplace=True)
		return train_valid		

	def manage_ticket( self, train_valid ):
		train_valid['Ticket'] = train_valid['Ticket'].apply( lambda x: self.extract_ticket( x ) )
		dummies_ticket = pd.get_dummies( train_valid['Ticket']  )
		train_valid['Ticket0'] = dummies_ticket[dummies_ticket.columns[0]] 
		train_valid['Ticket1'] = dummies_ticket[dummies_ticket.columns[1]] 
		train_valid.drop('Ticket', axis=1, inplace=True)
		return train_valid		
		
	def manage_age( self, train_valid ):
		train_valid['Age'] = train_valid['Age'].apply( lambda x: self.extract_age( x ) )
		return train_valid	

	def manageDatasets(self):

		le = LabelEncoder()
		
		path = "/home/andrea/Desktop/python/titanic/"
		train_data = path + "train.csv"
		test_data = path + "test.csv"

		CSV_COLUMNS = [ "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
		CSV_COLUMNS_TEST = [ "PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

		CSV_OUTPUT = ["PassengerId"]

		CSV_TRAIN = [ "PassengerId", "Age","Fare","Sex0","Sex1", "Pclass","SibSp", "Parch", "Cabin0",
			 "Cabin1", "Name0", "Name1"]
		CSV_TARGET = ["Survived"]
		
		train_valid = shuffle(pd.read_csv( train_data, names=CSV_COLUMNS, header=0, skipinitialspace=True))
		train_valid = train_valid.groupby(train_valid.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
		train_valid = self.manage_sex(train_valid)
		train_valid = self.manage_cabin(train_valid)
		train_valid = self.manage_name(train_valid)
		train_valid = self.manage_ticket(train_valid)
		train_valid = self.manage_age(train_valid)
		
		dataset_test = pd.read_csv( test_data, names=CSV_COLUMNS_TEST, header=0, skipinitialspace=True)
		dataset_test = dataset_test.groupby(dataset_test.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))

		
		dataset_test = self.manage_sex(dataset_test)
		dataset_test = self.manage_cabin(dataset_test)
		dataset_test = self.manage_name(dataset_test)
		dataset_test = self.manage_ticket(dataset_test)
		dataset_test = self.manage_age(dataset_test)

		self.X = train_valid[ CSV_TRAIN ]
		self.Y = train_valid[ CSV_TARGET ].values.ravel()

		self.X_test = dataset_test[ CSV_TRAIN ]
		self.X_output = dataset_test[ CSV_OUTPUT ]

	def predict( self, type ):
		y_pred_test = self.gbm.predict(self.X_test)

		y_pred_test = list( map( lambda p: self.rounder(p, 0.5) , y_pred_test ) )

		ids = self.X_test['PassengerId']
		d = {'PassengerId': ids , 'Survived': y_pred_test}
		self.test_results = pd.DataFrame(data=d)

	def run( self ):

		
		kfold = StratifiedKFold(n_splits=10, random_state=7)
		results = cross_val_score( self.gbm, self.X, self.Y, cv=kfold)
	
		return results.mean()




