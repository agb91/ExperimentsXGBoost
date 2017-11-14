from __future__ import division
from gene import Gene
import pandas as pd
from geneCreator import GeneCreator
from breeder import Breeder
from titanic_boost_regressor import TitanicBoostRegressor
from titanic_boost_classifier import TitanicBoostClassifier
from various_forests import VariousForests
from dataReader import DataReader

if __name__ == "__main__":
  
  population = 18
  nGenerations = 4

  creator = GeneCreator()
  breeder = Breeder()
  dataReader = DataReader()
  X,Y,X_test,X_output = dataReader.readData()

  #try regressors
  print( "\n\n\n########################## TRY! ##########################")
  generation = breeder.getFirstGeneration( population )
  generation = breeder.run( generation )

  for i in range ( 0 , nGenerations ):
    print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.getNewGeneration(generation , population)
    generation = breeder.run( generation )
    #print( "gen lenght: " + str(len(generation)) )
    best = breeder.takeBest( generation )
    #best.toStr()
    tot = 0
    for k in range( 0, len(generation) ):
      tot = tot + generation[i].level

    print("we reach a correctness percentage of: " + str( best.level) )
    print("we reach a medium result of: " + str( tot / len(generation)  ) )

  #nn = NeuralAbalone( confs )
  #loss = nn.run()
  print( "\n\n\n########################## IN THE END ##########################")
    
  print("we reach a correctness percentage of: " + str( best.level) )
  print( best.toStr() )


  runner = TitanicBoostClassifier() # just to initialize 
  if( best.way == 0 ):
    runner = TitanicBoostClassifier()
  if( best.way == 1 ):
    runner = TitanicBoostRegressor()
  if( best.way == 2 ):
    runner = VariousForests()  
  
  runner.setDatasets(X , Y , X_test , X_output)
  runner.set_gene_to_model( best )
  runner.run()
  runner.predict()
  df_test = runner.test_results

  print( "Ok, now I write the result;" )
  df_test.to_csv("toSubmit.csv", index=False)
