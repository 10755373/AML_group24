import pandas as pd
from functions import preprocessing

def load_data(test_file_path = './flights_test.csv', training_file_path = './flights_train.csv'):
    flights_test = pd.read_csv(test_file_path)
    flights_train = pd.read_csv(training_file_path)

    flights_train = flights_train.sort_values(['AIRLINE', 'FLIGHT_NUMBER']).fillna(method='backfill')
    flights_test = flights_test.sort_values(['AIRLINE', 'FLIGHT_NUMBER']).fillna(method='backfill')

    flights_train = preprocessing(flights_train)
    flights_test = preprocessing(flights_test)

    for category in flights_train.columns[flights_train.dtypes=='object']:
      flights_train[category] = pd.Categorical(flights_train[category])

    for category in flights_test.columns[flights_test.dtypes=='object']:
      flights_test[category] = pd.Categorical(flights_test[category])

    return (flights_test, flights_train)