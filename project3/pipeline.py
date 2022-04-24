from typing import Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from scipy import interpolate


class Pipeline:

    @staticmethod
    def format_data(data: pd.DataFrame, k: int, features: list[str], noise_prev_no=0) -> Tuple[np.ndarray, np.array]:
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
            cur_sequence = x[i-k: i].copy()
            cur_target = y[i-1]

            if noise_prev_no > 0:
                sigma = 0.1
                noise = sigma * np.random.randn(noise_prev_no)
                cur_sequence[:noise_prev_no, -1] += noise

            X_data[i-k, :, :] = cur_sequence.reshape(1, k, X_data.shape[2])
            y_data.append(cur_target)

        return X_data, np.asarray(y_data)

    @staticmethod
    def process(data: pd.DataFrame, altered=False) -> Tuple[pd.DataFrame, preprocessing.StandardScaler]:
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

        # Time features
        dt = df.start_time.apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df['time_of_day'] = dt.apply(lambda x: x.hour)
        df['time_of_week'] = dt.apply(lambda x: x.weekday())
        df['time_of_year'] = dt.apply(lambda x: x.month % 12 // 3)
        # Divide the hour into 0,1,..11
        df['time_of_hour'] = dt.apply(lambda x: (x.minute // 5))

        df['new_hour'] = dt.apply(lambda x: (x.minute == 0)).astype(np.float32)

        df['time_of_day_sin'] = np.sin(df['time_of_day'] * (2 * np.pi / 24))
        df['time_of_day_cos'] = np.cos(df['time_of_day'] * (2 * np.pi / 24))
        df['time_of_week_sin'] = np.sin(df['time_of_week'] * (2 * np.pi / 7))
        df['time_of_week_cos'] = np.cos(df['time_of_week'] * (2 * np.pi / 7))
        df['time_of_year_sin'] = np.sin(df['time_of_year'] * (2 * np.pi / 12))
        df['time_of_year_cos'] = np.cos(df['time_of_year'] * (2 * np.pi / 12))
        df['time_of_hour_sin'] = np.sin(df['time_of_hour'] * (2 * np.pi / 12))
        df['time_of_hour_cos'] = np.cos(df['time_of_hour'] * (2 * np.pi / 12))

        df['sum'] = df['total'] + df['flow']

        # Calculate structural imbalance
        tdf = df.copy(deep=True)
        n = len(tdf)
        start = (6-tdf.time_of_hour[0]) % 12
        x = np.arange(start, n, 12)
        tck = interpolate.splrep(x, tdf['sum'][x], s=1)
        xfit = np.arange(0, n)
        yfit = interpolate.splev(xfit, tck, der=0)

        tdf['smooth'] = yfit

        df['structural_imbalance'] = tdf['sum'] - tdf['smooth']

        if altered:  # Swap y variable for the altered forecasting task
            df['y'] = df['y'] - df['structural_imbalance']

        # Standard scale numerical features
        numerical_features = ['hydro', 'micro', 'thermal', 'wind', 'total', 'y',
                              'sys_reg', 'flow', 'structural_imbalance', 'sum']

        scaler = preprocessing.StandardScaler().fit(df[numerical_features])

        df[numerical_features] = scaler.transform(df[numerical_features])

        # Lag features
        last_day_offset = 24*60 // 5  # timesteps are 5 minutes
        last_week_offset = 7 * 24*60 // 5
        df['previous_y'] = df['y'].shift(1)
        df['previous_20y'] = df['y'].shift(20)
        df['prev_day_y'] = df['y'].shift(last_day_offset)
        df['prev_week_y'] = df['y'].shift(last_week_offset)

        # Remove the first rows which contains NaNs.
        df = df[last_week_offset:]
        return df, scaler
