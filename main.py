from gene import Gene
from geneCreator import GeneCreator
from breeder import Breeder

if __name__ == "__main__":
  
  population = 60
  nGenerations = 30

  creator = GeneCreator()
  breeder = Breeder()
  generation = breeder.getFirstGeneration( population )
  generation = breeder.run( generation , 1 )

  for i in range ( 0 , nGenerations ):
    print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.getNewGeneration(generation , population)
    generation = breeder.run( generation , 1 )
    best = breeder.takeBest( generation )
    #best.toStr()
    print("we reach a correctness percentage of: " + str( best.level) )

  #nn = NeuralAbalone( confs )
  #loss = nn.run()
  print( "\n\n\n########################## IN THE END ##########################")
    
  print("we reach a correctness percentage of: " + str( best.level) )
  print( best.toStr() )
