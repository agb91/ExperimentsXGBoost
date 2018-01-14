from __future__ import division
from gene import Gene
import pandas as pd
from gene_creator import GeneCreator
from breeder import Breeder
from titanic_boost_regressor import TitanicBoostRegressor
from titanic_boost_classifier import TitanicBoostClassifier
from various_forests import VariousForests
from data_reader import DataReader
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt  


class FieldsSelector:

	def __init__( self, X, Y ):
		self.X = X
		self.Y = Y

	def get_features( self ):
		clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
		clf = clf.fit( self.X, self.Y )

		features = pd.DataFrame()
		features['feature'] = X.columns
		features['importance'] = clf.feature_importances_

		features.sort_values(by=['importance'], ascending=False, inplace=True)
		features = features.head( n = 12 )

		return features["feature"]



