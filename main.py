from __future__ import division
from gene import Gene
import pandas as pd
from gene_creator import GeneCreator
from breeder import Breeder
from titanic_boost_regressor import TitanicBoostRegressor
from titanic_boost_classifier import TitanicBoostClassifier
from various_forests import VariousForests
from data_reader import DataReader


if __name__ == "__main__":
  
  population = 80
  n_generations = 15

  creator = GeneCreator()
  breeder = Breeder()
  data_reader = DataReader()
  X,Y,X_test,X_output = data_reader.read_data()

  #print( X.head() )

  #try regressors
  print( "\n\n\n########################## TRY! ##########################")
  generation = breeder.get_first_generation( population )
  generation = breeder.run( generation )

  for i in range ( 0 , n_generations ):
    print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.get_new_generation(generation , population)
    generation = breeder.run( generation )
    #print( "gen lenght: " + str(len(generation)) )
    best = breeder.take_best( generation )
    #best.to_str()
    tot = 0
    string_ways = str("")
    for k in range( 0, len(generation) ):
      string_ways += str( generation[k].way ) + str("-")
      tot = tot + generation[k].level

    print("we reach a correctness percentage of: " + str( best.level) +" using way: " + str( best.way ))
    print("we reach a medium result of: " + str( tot / len(generation)  ) )
    print( "ways: " + str( string_ways ) )
  #nn = NeuralAbalone( confs )
  #loss = nn.run()
  print( "\n\n\n########################## IN THE END ##########################")
    
  print("we reach a correctness percentage of: " + str( best.level) + 
    " using way: " + str( best.way ))
  print( best.to_str() )


  runner = TitanicBoostClassifier() # just to initialize 
  if( best.way == 0 ):
    print(" to predict I set boost classifier ")
    runner = TitanicBoostClassifier()
  else:
    if( best.way == 1 ):
      print(" to predict I set boost Regressor ")
      runner = TitanicBoostRegressor()
    else:
      print( "to predict I set some forests, number: " + str(best.way) )
      runner = VariousForests()  
  
  runner.set_datasets(X , Y , X_test , X_output)
  runner.set_gene_to_model( best )
  runner.run()
  runner.predict()
  df_test = runner.test_results

  print( "Ok, now I write the result;" )
  df_test.to_csv("toSubmit.csv", index=False)
