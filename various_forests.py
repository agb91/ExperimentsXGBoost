import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from gene import Gene

class VariousForests:

	def __init__( self  ):
		pass

	def setDatasets( self , X, Y, X_test, X_output ):
		self.X = X
		self.Y = Y
		self.X_test = X_test
		self.X_output = X_output
		self.test_results = None
		self.way = None
		self.gene = None
		self.runner = None
	
	def set_gene_to_model( self, gene ):
		self.way = gene.way
		self.max_depth = gene.max_depth
		self.subsample = gene.subsample
		self.learning_rate = gene.learning_rate
		self.n_estimators = gene.n_estimators
		self.n_neighbors = gene.n_neighbors
		self.gene = gene


	def predict( self ):
		
		self.runner.fit(self.X, self.Y)
		y_pred_test = self.bestRunner.predict(self.X_test)

		ids = self.X_output['PassengerId']
		d = {'PassengerId': ids , 'Survived': y_pred_test}
		self.test_results = pd.DataFrame(data=d)


	def run( self ):
		random_state = 2
		kfold = StratifiedKFold(n_splits=5)

		if( self.way == 2 ):
			self.runner = SVC(random_state=random_state)
		
		if( self.way == 3 ):
			self.runner = DecisionTreeClassifier(max_depth = self.max_depth, random_state=random_state)
		
		if( self.way == 4 ):
			self.runner = AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
				random_state=random_state, learning_rate=(float(self.learning_rate)*5.0), 
				n_estimators = self.n_estimators)
		
		if( self.way == 5 ):
			self.runner = GradientBoostingClassifier(random_state=random_state, 
				learning_rate = self.learning_rate, n_estimators = self.n_estimators 
				,max_depth = self.max_depth, subsample = self.subsample )
		
		if( self.way == 6 ):
			self.runner = KNeighborsClassifier( n_neighbors = self.n_neighbors )


		if( self.way == 7 ):
			nest = int( float(self.n_estimators) / 10.0 )
			self.runner = RandomForestClassifier(random_state=random_state, n_estimators= nest,
				max_depth = self.max_depth, )
		
		#or the more basic:
		if( self.way == 8 ):
			self.runner = RandomForestClassifier( n_estimators=100 )

		results = cross_val_score( self.runner, self.X, self.Y, scoring = "accuracy", 
				cv = kfold)
		thisResult = results.mean()
		
		return thisResult




