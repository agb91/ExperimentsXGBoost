from  __future__ import division
from gene import Gene
import random

'''
	the following class creates randomly a Gene that contains all the needed configurations.
	Each attribute has a range that is limited. The limits are chosen by me according to my personal
	experience 
'''

class GeneCreator:
	
	def random_col(self):
		result = (random.random() / 2) + 0.5  # I want something in the range 0.5 - 1
		return ( result )

	def random_way( self ):
		return ( random.randint(0,8) )

	def random_sample(self):
		result = (random.random() / 2) + 0.5  # I want something in the range 0.5 - 1
		return ( result )

	def random_learning(self):
		result = (random.random() / 600.0) + 0.01   # I want something in the range 0.01 - 0.2
		return ( result )


	def random_estimators(self):
		return ( random.randint(1,1000) )
	
	def random_depth_weight(self):
		return ( random.randint(3,7) )

	def random_child_weight(self):
		return ( random.randint(1,8) )	

	def random_create(self):
		
		col_by_tree = self.random_col()
		subsample = self.random_sample()
		min_child_weight = self.random_child_weight()
		max_depth = self.random_depth_weight()
		n_estimators = self.random_estimators()
		learning_rate = self.random_learning()
		way = self.random_way()
		n_neighbors = self.random_child_weight()	
		gene = Gene( col_by_tree, subsample, min_child_weight, max_depth, n_estimators, 
			learning_rate , way, n_neighbors)
		return gene 	
	