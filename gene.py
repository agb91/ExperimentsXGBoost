import random

class Gene:


	def __init__( self, col_by_tree, subsample, min_child_weight, max_depth, n_estimators, 
		learning_rate, way, n_neighbors ):
		self.col_by_tree = col_by_tree
		self.subsample = subsample
		self.min_child_weight = min_child_weight
		self.max_depth = max_depth
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.way = way
		self.n_neighbors = n_neighbors
		self.level = None


	def to_str( self ):
		print( "gene: \ncbt: " + str( self.col_by_tree ) + " -- subs" + str( self.subsample )
			+ " -- mcw" + str( self.min_child_weight ) + "--    max_depth: " 
			+ str( self.max_depth )
			+ "-- nest " + str( self.n_estimators ) + "-- way: " + str( self.way ) 
			+ "-- neighbs  " + str( self.n_neighbors ) + ";   level: " + str( self.level ) )	

	def set_fitness_level( self, l ):
		self.level = l	

