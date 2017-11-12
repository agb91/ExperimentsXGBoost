from gene import Gene
from geneCreator import GeneCreator
from titanic_boost_regressor import TitanicBoostRegressor
import numpy as np
import random
import math


class Breeder:

	def getNewGeneration( self, old, n):
		geneCreator = GeneCreator()
		newGeneration = list()
		strongestN = 5
		if(n<5):
			strongestN = n
		goods = self.takeGoods( old , strongestN )
		#I want to maintain old goods in my genetic pools
		for i in range( 0, len(goods) ):
			newGeneration.append(goods[i])
		randomAdds = math.ceil(n/3) 
		#I want some sons generated by goods
		for i in range( 0 , (n - strongestN - randomAdds ) ):
			son = self.getSon( goods )
			newGeneration.append(son)
		#I want also some randoms new borns
		for i in range( 0, randomAdds ):
			newGeneration.append( geneCreator.randomCreate() )
		
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

	def run(self, generation, runner):
		runnedGeneration = list()
		
		for i in range( 0 , len(generation)):
			thisGene = generation[i]
			tb = runner
			tb.set_gene_to_model( thisGene )
			thisGene.setFitnessLevel( tb.run() ) 
			runnedGeneration.append(thisGene)
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
		genesSet = set(genes)
		genes = list( genesSet ) # no doubles!
		result = sorted(genes, key=lambda x: x.level, reverse=True)
		
		#for i in range( 0, len(result) ):
		#	print( result[i].level )		
		
		return result

	def takeGoods( self, genes, n ):
		goods = []

		for i in range(0, len(genes) ):
			g = genes[i]
			goods.append(g)
			goods = self.orderGenes( goods )
			if( len( goods ) > n):
				goods = goods[ 0 : n ]

		#for i in range( 0, len(goods) ):
		#	print( goods[i].level )		
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