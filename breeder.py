from gene import Gene
from geneCreator import GeneCreator
from titanic_boost import TitanicBoost
import numpy as np
import random


class Breeder:

	def getNewGeneration( self, old, n):
		newGeneration = list()
		strongestN = 3
		if(n<3):
			strongestN = n
		goods = self.takeGoods( old , strongestN )
		for i in range( 0 , (n - 1 ) ):
			son = self.getSon( goods )
			newGeneration.append(son)
		oldBest = self.takeBest(old)	
		newGeneration.append( oldBest )	
		return newGeneration

	def getSon( self, parents ):
		
		cbti = random.randint(0, (len(parents) - 1 ) )
		cbt = parents[cbti].col_by_tree
		
		ssi = random.randint(0, (len(parents) - 1 ))
		ss = parents[ssi].subsample

		mcwi = random.randint(0, (len(parents) - 1 ))
		mcw = parents[mcwi].min_child_weight

		mdi = random.randint(0, (len(parents) - 1 ))
		md = parents[mdi].max_depth

		nei  = random.randint(0, (len(parents) - 1 ))
		ne = parents[nei].n_estimators

		lri = random.randint(0, (len(parents) - 1 ))
		lr = parents[lri].learning_rate

		son = Gene( cbt, ss, mcw, md, ne, lr )
		return son	

	def run(self, generation, verbose):
		runnedGeneration = list()
		
		for i in range( 0 , len(generation)):
			g = generation[i]
			tb = TitanicBoost( g )
			level = 0
			for k in range( 0 , 2 ):
				level = level + tb.run()
			level = level / 2.0	
			g.setFitnessLevel( level ) 
			#g.toStr()
			runnedGeneration.append(g)
		return runnedGeneration	

	def getFirstGeneration( self, n ):
		genes = list()
		creator = GeneCreator()
		for i in range( 0 , n):
			g = creator.randomCreate()
			genes.append(g)
		return genes

	def orderGenes( self , genes ):
		result = []
		result = sorted(genes, key=lambda x: x.level, reverse=True)
		return result

	def takeGoods( self, genes, n ):
		goods = []

		for i in range(0, len(genes) ):
			g = genes[i]
			goods.append(g)
			goods = self.orderGenes( goods )
			if( len( goods ) > n):
				goods = goods[ 0 : n ]
		return goods		    

	def takeBest( self, genes ):

		maxLevel = 0 #level of correctness percentage
		bestGene = None

		for i in range(0, len(genes) ):
			g = genes[i]
			if( g.level > maxLevel ):
				bestGene = g
				maxLevel = g.level

		return bestGene		