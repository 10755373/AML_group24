import datetime
import numpy as np

def aggregate_date_time(values):
    if(len(values) == 5):
        year, month, day, time, departure = values
        year = int(year)
        month = int(month)
        day = int(day)
        time = int(time)

        hour, minutes = int(time/100), time % 100
        if(hour == 24):
            hour = 0
        dateTime = datetime.datetime(year, month, day, hour, minutes)
        if(dateTime < departure and (departure-dateTime).total_seconds()/60 > 50):
            return dateTime + datetime.timedelta(days=1)
        return dateTime

    year, month, day, time = [int(x) for x in values]
    hour, minutes = int(time/100), time % 100
    if(hour == 24):
        hour = 0
    return datetime.datetime(year, month, day, hour, minutes)


def aggregate_time_in_minutes(dateTime):
    return dateTime.hour*60 + dateTime.minute


def preprocessing(x):
    x['scheduled_departure_date_time'] = x[['YEAR', 'MONTH', 'DAY',
                                            'SCHEDULED_DEPARTURE']].apply(lambda x: aggregate_date_time(x), axis=1)
    x['scheduled_departure_date'] = x['scheduled_departure_date_time'].apply(
        lambda x: x.date())
    x['scheduled_departure_timestamp'] = x['scheduled_departure_date_time'].apply(
        lambda x: x.timestamp())
    # x['scheduled_arrival_date_time'] = x[['YEAR', 'MONTH', 'DAY', 'SCHEDULED_ARRIVAL']].apply(lambda x: aggregate_date_time(x), axis=1)
    x['departure_date_time'] = x[['YEAR', 'MONTH', 'DAY', 'DEPARTURE_TIME',
                                  'scheduled_departure_date_time']].apply(lambda x: aggregate_date_time(x), axis=1)
    x['departure_timestamp'] = x['departure_date_time'].apply(
        lambda x: x.timestamp())
    # x['scheduled_departure_minutes'] = x['scheduled_departure_date_time'].apply(lambda x: aggregate_time_in_minutes(x))
    x['initial_delay'] = (x['departure_date_time'] -
                          x['scheduled_departure_date_time']).apply(lambda x: x.total_seconds() / 60)

    # aggregate mean of initial_delay during day and origin_airport
    delay_at_origin_features = ['ORIGIN_AIRPORT', 'scheduled_departure_date']
    groupedX = x.groupby(delay_at_origin_features).agg(
        {'initial_delay': ['sum', 'mean']})
    groupedX.columns = ['_'.join(column) for column in groupedX.columns]
    x = x.join(groupedX, on=delay_at_origin_features)
    return x


def mse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual, pred)).mean()
