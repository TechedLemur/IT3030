from typing import Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing


class Pipeline:

    @staticmethod
    def format_data(data: pd.DataFrame, k: int, features: list[str], drop_prev_no=0) -> Tuple[np.ndarray, np.array]:
        '''
        input:
            data - the pandas dataframe of (n, p+1) shape, where n is the number of rows,
                p+1 is the number of predictors + 1 target column
            k    - the length of the sequence, namely, the number of previous rows 
                (including current) we want to use to predict the target.
        output:
            X_data - the predictors numpy matrix of (n-k, k, p) shape
            y_data - the target numpy array of (n-k, 1) shape
        '''
        # initialize zero matrix of (n-k, k, p) shape to store the n-k number
        # of sequences of k-length and zero array of (n-k, 1) to store targets
        x = data[features].to_numpy()
        y = data['y'].to_numpy()

        X_data = np.zeros([x.shape[0]-k, k, x.shape[1]])
        y_data = []

        # run loop to slice k-number of previous rows as 1 sequence to predict
        # 1 target and save them to X_data matrix and y_data list
        for i in range(k, x.shape[0]):
            cur_sequence = x[i-k: i]
            cur_target = y[i-1]

            if drop_prev_no > 0:
                cur_sequence[-drop_prev_no:, -1] = 0

            X_data[i-k, :, :] = cur_sequence.reshape(1, k, X_data.shape[2])
            y_data.append(cur_target)

        return X_data, np.asarray(y_data)

    @staticmethod
    def process(data: pd.DataFrame) -> Tuple[pd.DataFrame, preprocessing.StandardScaler]:
        """
        Process the dataframe. Remove outliers, scale etc.
        Returns the transformed dataframe along with a scaler to inverse scale the target.
        """
        df = data.copy(deep=True)

        df['flow'] = -df['flow']  # Flip sign because of mistake in dataset

        df.loc[df['y'] < -5000, 'y'] = 5.687162  # Set wrong values to mean
        df.loc[df['y'] > 5000, 'y'] = 5.687162  # Set wrong values to mean
        # TODO: CLip more values
        df = df.drop(columns='river')  # Drop useless column

        numerical_features = ['hydro', 'micro', 'thermal', 'wind', 'total', 'y',
                              'sys_reg', 'flow']

        scaler = preprocessing.StandardScaler().fit(df[numerical_features])

        df[numerical_features] = scaler.transform(df[numerical_features])

        dt = df.start_time.apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df['time_of_day'] = dt.apply(lambda x: x.hour)
        df['time_of_week'] = dt.apply(lambda x: x.weekday())
        df['time_of_year'] = dt.apply(lambda x: x.month % 12 // 3)
        df['new_hour'] = dt.apply(lambda x: (x.minute == 0)).astype(np.float32)

        df['time_of_day_sin'] = np.sin(df['time_of_day'] * (2 * np.pi / 23))
        df['time_of_day_cos'] = np.cos(df['time_of_day'] * (2 * np.pi / 23))
        df['time_of_week_sin'] = np.sin(df['time_of_week'] * (2 * np.pi / 6))
        df['time_of_week_cos'] = np.cos(df['time_of_week'] * (2 * np.pi / 6))
        df['time_of_year_sin'] = np.sin(df['time_of_year'] * (2 * np.pi / 11))
        df['time_of_year_cos'] = np.cos(df['time_of_year'] * (2 * np.pi / 11))

        last_day_offset = 24*60 // 5  # timesteps are 5 minutes
        last_week_offset = 7 * 24*60 // 5
        df['previous_y'] = df['y'].shift(1)
        df['prev_day_y'] = df['y'].shift(last_day_offset)
        df['prev_week_y'] = df['y'].shift(last_week_offset)
        # Remove the first rows which contains NaNs.
        df = df[last_week_offset:]
        return df, scaler
