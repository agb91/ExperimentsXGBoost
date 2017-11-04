import random

class Gene:


	def __init__( self, col_by_tree, subsample, min_child_weight, max_depth, n_estimators, 
		learning_rate ):
		self.col_by_tree = col_by_tree
		self.subsample = subsample
		self.min_child_weight = min_child_weight
		self.max_depth = max_depth
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.level = None

	def toStr( self ):
		print( "gene: \ncbt: " + str( self.col_by_tree ) + " -- subs" + str( self.subsample )
			+ " -- mcw" + str( self.min_child_weight ) + "--    max_depth: " + str( self.max_depth )
			+ "-- nest " + str( self.n_estimators ) + ";   level: " + str( self.level ) )	

	def setFitnessLevel( self, l ):
		self.level = l	

	def isAcceptable( self ):
		if(self.col_by_tree < 0 or self.col_by_tree>1 ):
			return False
		if( self.subsample < 0 or self.subsample>1 ):
			return False	
		if( self.learning_rate < 0 or self.learning_rate > 1 ):
			return False		
		if( self.min_child_weight < 1 or  self.min_child_weight > 10 ):
			return False	
		if( self.max_depth < 1 or  self.max_depth > 10 ):
			return False	
		if( self.n_estimators < 1 or  self.n_estimators > 5000 ):
			return False			
		return True	

