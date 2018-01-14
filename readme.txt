Titanic - Machine Learning

This demo project is based on Python3, it aims to analyze the database of the victims of the Titanic's shipwreck happened in 1912.
The system read data related to passenger's name, ticket number, class, age etc from a csv file (using Pandas), operate some data wrangling in order to create more composite information and uses this information to train different machine learning algorithms ( SVC, Decision Tree, AdaBoost, K-Neighbors, Random Forest) with different sets of parameters. The various algorithm are implemented using Sklearn and xgBoost libraries.

The system use a genetic algorithm to efficiently optimize the parameters and choose the best algorithm, with a fitness function based on stratified crossvalidation evalution.

The project is runnable using:
python3 main.py

I've tried to partecipare at the Kaggle competition related on this database ( reachable here https://www.kaggle.com/c/titanic ) abtaining a top 10% position (696 on 9000+)
