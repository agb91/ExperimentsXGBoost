from gene import Gene
import random

class GeneCreator:
	
	def randomColSample(self):
		result = (random.random() / 120.0) + 0.2  # I wanna something not so near to zero..
		return ( result )

	def randomLearning(self):
		result = (random.random() / 200.0) + 0.01   # I wanna something around 0.01 - 0.5
		return ( result )


	def randomEstimators(self):
		return ( random.randint(1,1000) )
	
	def randomDepthWeight(self):
		return ( random.randint(1,7) )

	def randomCreate(self):
		
		col_by_tree = self.randomColSample()
		subsample = self.randomColSample()
		min_child_weight = self.randomDepthWeight()
		max_depth = self.randomDepthWeight()
		n_estimators = self.randomEstimators()
		learning_rate = self.randomLearning()
		
		gene = Gene( col_by_tree, subsample, min_child_weight, max_depth, n_estimators, 
			learning_rate )
		return gene 	
	