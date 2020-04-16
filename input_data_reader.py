import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class InputDataReader:

    def __init__(self):
        self.data = None

    def read_load_pattern_data(self, path='loaddata.csv'):

        if not os.path.exists('load_df.csv'):
            csv_data = pd.read_csv(path, names=['CustomerID'] + list(np.arange(0, 24, 0.5)), index_col=1)
            csv_data.to_csv('load_df.csv', index=False)
        else:
            csv_data = pd.read_csv('load_df.csv')

        self.data = csv_data

    def read_load_time_series_data(self, path='loaddata.csv'):
        if not os.path.exists('pivoted_loaddata.csv'):
            csv_data = pd.read_csv(path, names=['CustomerID'] + list(np.arange(0, 24, 0.5)), index_col=1)
            csv_data = csv_data.drop(csv_data.index[0])
            df = csv_data.melt(id_vars='CustomerID', var_name='Timestamp', value_name='Energy')
            reduced_df = pd.DataFrame(columns=['CustomerID', 'Timestamp', 'Energy'])

            # grouped = df.groupby(by=['CustomerID', 'Timestamp'])
            #
            # for name, group in grouped:
            #     df.loc[(df.CustomerID == name[0]) & (df.Timestamp == name[1]), 'Timestamp'] \
            #         = pd.date_range(start=pd.to_datetime(name[1]), freq='D', periods=len(group.Timestamp.index))

            customer_ids = df.CustomerID.unique()
            timestamps = df.Timestamp.unique()
            timestamp_origin = pd.Timestamp(datetime.now())

            for cid in customer_ids:
                cid_arr = (df.CustomerID == cid)
                if cid == 1001:
                    break
                for ts in timestamps:
                    bool_array = cid_arr & (df.Timestamp == ts)

                    df.loc[bool_array, 'Timestamp'] = \
                        pd.date_range(start=timestamp_origin + pd.Timedelta(hours=ts),
                                      freq='D', periods=np.count_nonzero(bool_array))

                    reduced_df = reduced_df.append(df.loc[bool_array])

            reduced_df['Month'] = reduced_df.Timestamp.dt.month
            reduced_df['DayOfWeek'] = reduced_df.Timestamp.dt.dayofweek
            reduced_df['Hour'] = reduced_df.Timestamp.dt.hour
            reduced_df.to_csv('pivoted_loaddata.csv', index=False)

        else:
            reduced_df = pd.read_csv('pivoted_loaddata.csv')

        self.data = reduced_df
