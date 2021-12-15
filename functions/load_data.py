import numpy as np

def loaddata():
    # read al necessary data 

    airports = pd.read_csv('./data/airports.csv', sep=',')
    airlines = pd.read_csv('./data/airlines.csv')

    # data to test and train on
    flights_test = pd.read_csv('./data/flights_test.csv')
    flights_train = pd.read_csv('./data/flights_train.csv')

    # format on how to submit you results
    submit_sample = pd.read_csv('./data/submit_sample.csv')

    return airports, airlines, flights_train, flights_test, submit_sample