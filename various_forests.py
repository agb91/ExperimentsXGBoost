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
		self.bestRunner = None

	def set_gene_to_model( self, gene ):
		pass	

	def predict( self ):
		
		self.bestRunner.fit(self.X, self.Y)
		y_pred_test = self.bestRunner.predict(self.X_test)

		ids = self.X_test['PassengerId']
		d = {'PassengerId': ids , 'Survived': y_pred_test}
		self.test_results = pd.DataFrame(data=d)


	def run( self ):
		random_state = 2
		kfold = StratifiedKFold(n_splits=5, random_state=random_state)

		names_cl = list()
		classifiers = list()
		classifiers.append(SVC(random_state=random_state))
		names_cl.append( "SVC" )
		classifiers.append(DecisionTreeClassifier(random_state=random_state))
		names_cl.append("DecisionTree")
		classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
		names_cl.append("AdaBoost")
		classifiers.append(RandomForestClassifier(random_state=random_state))
		names_cl.append("RandomForest")
		classifiers.append(GradientBoostingClassifier(random_state=random_state))
		names_cl.append("GradientBoosting")
		classifiers.append(KNeighborsClassifier())
		names_cl.append("KNeighbors")
	
		cv_results = []
		bestResult = 0.0
		bestName = None
		for i in range(0, len( classifiers ) ) :
			results = cross_val_score(classifiers[i], self.X, self.Y, scoring = "accuracy", 
				cv = kfold)
			thisResult = results.mean()
			thisName = names_cl[i]
			if( thisResult > bestResult):
				bestResult = thisResult
				bestName = thisName
				self.bestRunner = classifiers[i]
		return bestResult




