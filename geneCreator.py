from gene import Gene
import random

class GeneCreator:
	
	def randomCol(self):
		result = (random.random() / 2) + 0.5  # I wanna something in 0.5 - 1
		return ( result )

	def randomWay( self ):
		return ( random.randint(0,1) )

	def randomSample(self):
		result = (random.random() / 2) + 0.5  # I wanna something in 0.5 - 1
		return ( result )

	def randomLearning(self):
		result = (random.random() / 600.0) + 0.01   # I wanna something around 0.01 - 0.2
		return ( result )


	def randomEstimators(self):
		return ( random.randint(1,1000) )
	
	def randomDepthWeight(self):
		return ( random.randint(3,10) )

	def randomChildWeight(self):
		return ( random.randint(1,8) )	

	def randomCreate(self):
		
		col_by_tree = self.randomCol()
		subsample = self.randomSample()
		min_child_weight = self.randomChildWeight()
		max_depth = self.randomDepthWeight()
		n_estimators = self.randomEstimators()
		learning_rate = self.randomLearning()
		way = self.randomWay()
		
		gene = Gene( col_by_tree, subsample, min_child_weight, max_depth, n_estimators, 
			learning_rate , way)
		return gene 	
	