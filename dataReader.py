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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

class DataReader:

	def __init__( self  ):
		pass



	def extract_name(self , name):
		part = name.split(',')[1].split('.')[0].strip()

		if( str(part) == 'Don' or str(part) == 'Rev' or str(part) == 'Jonkheer' 
			or str(part) == 'Capt'):
			return 0
 
		if( str(part) == 'Mr' ):
			return 1

		if( str(part) == 'Dr' ):
			return 2

		if( str(part) == 'Col' or str(part) == 'Major' ):
			return 3

		if( str(part) == 'Dr' ):
			return 4

		if( str(part) == 'Miss' ):
			return 5
	
		if( str(part) == 'Mrs' ):
			return 6
		
		if( str(part) == 'Mme' or str(part) == 'Ms' or str(part) == 'Mlle' 
			or str(part) == 'Sir' or str(part) == 'Lady' or str(part) == 'the Countess'):
			return 7

		return part


	def extract_ticket( self, ticket ):
		ticket = str(ticket)[:1].strip()
		if( ticket.isdigit() ):
			return 0
		else:
			return ticket

	def extract_crew( self, fare ):
		if( float (fare) == 0.0  ):
			return 1
		else:
			return 0			

	def extract_parch( self, parch ):
		if( parch == 6 or parch == 4 ):
			return 0
		if( parch == 5 ):
			return 1
		if( parch == 0 ):
			return 2
		if( parch == 2 ):
			return 3
		if( parch == 1 ):
			return 4
		if( parch == 3 ):
			return 5		
		return parch	

	def extract_sibsp( self, sibsp ):
		if( sibsp == 5 or sibsp == 8 ):
			return 0
		if( sibsp == 4 ):
			return 1
		if( sibsp == 3 ):
			return 2
		if( sibsp == 0 ):
			return 3
		if( sibsp == 2 ):
			return 4
		if( sibsp == 1 ):
			return 5		
		return sibsp	

	def extract_alone( self, family ):
		if( family == 0):
			return 1
		else:
			return 0			

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

	def manage_crew( self, train_valid ):
		train_valid['Crew'] = train_valid['Fare'].apply( lambda x: self.extract_crew( x ) )
		return train_valid	

	def manage_parch( self, train_valid ):
		train_valid['Parch'] = train_valid['Parch'].apply( lambda x: self.extract_parch( x ) )
		train_valid = pd.get_dummies(train_valid, columns = ["Parch"])
		return train_valid	

	def manage_sibsp( self, train_valid ):
		train_valid['SibSp'] = train_valid['SibSp'].apply( lambda x: self.extract_sibsp( x ) )
		train_valid = pd.get_dummies(train_valid, columns = ["SibSp"])
		return train_valid	

	def manage_is_alone( self, train_valid ):
		train_valid['Alone'] = train_valid['Parch'] + train_valid['SibSp']	
		train_valid['Alone'] = train_valid['Alone'].apply( lambda x: self.extract_alone( x ) )
		return train_valid

	def get_features( self,X,Y,n_features ):
		clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
		clf = clf.fit( X, Y )

		features = pd.DataFrame()
		features['feature'] = X.columns
		features['importance'] = clf.feature_importances_

		features.sort_values(by=['importance'], ascending=False, inplace=True)
		features = features.head( n = n_features )

		return features["feature"]

	def readData(self):
		le = LabelEncoder()
		
		path = "/home/andrea/Desktop/python/titanic/"
		train_data = path + "train.csv"
		test_data = path + "test.csv"

		CSV_COLUMNS = [ "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
		CSV_COLUMNS_TEST = [ "PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

		CSV_OUTPUT = ["PassengerId"]

		CSV_TARGET = ["Survived"]
		
		train_valid = shuffle( pd.read_csv( train_data, names=CSV_COLUMNS, header=0, skipinitialspace=True) )
		dataset_test = shuffle( pd.read_csv( test_data, names=CSV_COLUMNS_TEST, header=0, skipinitialspace=True) )
		
		global_dataset = pd.concat( [train_valid, dataset_test] )

		global_dataset = self.manage_sex(global_dataset)
		global_dataset = self.manage_cabin(global_dataset)
		global_dataset = self.manage_name(global_dataset)
		global_dataset = self.manage_ticket(global_dataset)
		global_dataset = self.manage_age(global_dataset)
		global_dataset = self.manage_embarked( global_dataset )
		global_dataset = self.manage_crew( global_dataset )
		global_dataset = self.manage_is_alone( global_dataset )
		global_dataset = self.manage_parch( global_dataset )
		global_dataset = self.manage_sibsp( global_dataset )



		global_dataset = global_dataset.groupby(global_dataset.columns, axis = 1).transform(
			lambda x: x.fillna(x.median()))

		global_dataset["Age"] = global_dataset["Age"].fillna( global_dataset["Age"].mean() )
		Y = train_valid[ CSV_TARGET ].values.ravel()

		#print( global_dataset.columns.values )

		X = global_dataset.head( 891 )
		X.drop("PassengerId", axis=1, inplace=True)
		X.drop("Survived", axis=1, inplace=True)
		
		X_test = global_dataset.tail( 418 )
		X_test.drop("Survived", axis=1, inplace=True)

		X_output = X_test[ CSV_OUTPUT ]
		X_test.drop("PassengerId", axis=1, inplace=True)
		
		features = self.get_features( X , Y , 12)
		X = X[ features ]
		X_test = X_test[features]
		return X,Y,X_test,X_output
		

