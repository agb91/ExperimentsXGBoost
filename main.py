from __future__ import division
from gene import Gene
import pandas as pd
from geneCreator import GeneCreator
from breeder import Breeder
from titanic_boost import TitanicBoost

if __name__ == "__main__":
  
  population = 14
  nGenerations = 6

  creator = GeneCreator()
  breeder = Breeder()
  

  #try regressors
  print( "\n\n\n########################## CLASSIFIERS ##########################")
  generation = breeder.getFirstGeneration( population )
  runner = TitanicBoost()
  runner.manageDatasets()
  generation = breeder.run( generation , runner)

  for i in range ( 0 , nGenerations ):
    print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.getNewGeneration(generation , population)
    generation = breeder.run( generation , runner)
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
    
  print("we reach with CLASSIFIERS a correctness percentage of: " + str( best.level) )
  print( best.toStr() )

  runner.set_gene_to_model( best )
  runner.predict()
  df_test = runner.test_results

  df_test.to_csv("toSubmit.csv", index=False)
