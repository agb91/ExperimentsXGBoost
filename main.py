from __future__ import division
from gene import Gene
import pandas as pd
from gene_creator import GeneCreator
from breeder import Breeder
from titanic_boost_regressor import TitanicBoostRegressor
from titanic_boost_classifier import TitanicBoostClassifier
from various_forests import VariousForests
from data_reader import DataReader

'''
this application is just a demo, but it can reach quite good results (actually the dataset is quite
little and not so complex). Tha system adopts a genetic algorithm (info here
 https://en.wikipedia.org/wiki/Genetic_algorithm) in order to find the best possible 
configuration to run some algorithms of machine learning to solve the problem exposed here:
https://www.kaggle.com/c/titanic

'''
if __name__ == "__main__":
  
  ## the application is quite fast, because the Db is small.. so in a normal laptop we can run with
  ## 15-20 generations and a population of 60-100..
  population = 80
  n_generations = 15

  # this class can create randome genes in order to initialize the genetic algoritgm
  creator = GeneCreator()

  # method to implement the genetic algorithm itself 
  breeder = Breeder()
  data_reader = DataReader.getInstance()
  X,Y,X_test,X_output = data_reader.read_data()

  #print( X.head() )

  #try regressors
  print( "\n\n\n########################## BEGIN! ##########################")
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

    print("the best result has percentage of correctness of : " + str( best.level) +" using the algorithm number: " + str( best.way ))
    print("we reach a medium result of: " + str( tot / len(generation)  ) )
    #print( "ways: " + str( string_ways ) )


  #loss = nn.run()
  print( "\n\n\n########################## IN THE END ##########################")
    
  print("the best result at the end has percentage of correctness of: " + str( best.level) + 
    " using the algorithm: " + str( best.way ))
  print( best.to_str() )


  runner = None # just to initialize 
  if( best.way == 0 ):
    print(" to predict I set xgboost classifier ")
    runner = TitanicBoostClassifier()
  else:
    if( best.way == 1 ):
      print(" to predict I set xgboost Regressor ")
      runner = TitanicBoostRegressor()
    else:
      print( "to predict I set some forest, number: " + str(best.way) )
      runner = VariousForests()  
  
  runner.set_datasets(X , Y , X_test , X_output)
  runner.set_gene_to_model( best )
  runner.run()
  runner.predict()
  df_test = runner.test_results

  print( "Ok, now I write the result on toSubmit.csv" )
  df_test.to_csv("toSubmit.csv", index=False)
