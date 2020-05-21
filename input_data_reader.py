import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class InputDataReader:

    def __init__(self):
        self.data = None

    def read_load_pattern_data(self, path='loaddata.csv'):

        # read in the csv data
        csv_data = pd.read_csv(path, names=['CustomerID'] + list(np.arange(0, 24, 0.5)), index_col=1)

        self.data = csv_data

    def read_load_time_series_data(self, path='loaddata.csv', customer_ids=[]):
        """
        convert load pattern data to database with columns: Customer Id, Timestamp and Energy Value
        """
        reread_csv = False
        # if the modified load pattern to load time series file has already been made, read it
        if os.path.exists('pivoted_loaddata.csv'):
            reduced_df = pd.read_csv('pivoted_loaddata.csv')
            # check if it contains the customer ids we need, if not read the ids needed
            if not reduced_df.CustomerID.isin(customer_ids).all():
                reread_csv = True
        else:
            reread_csv = True

        if reread_csv:
            csv_data = pd.read_csv(path, names=['CustomerID'] + list(np.arange(0, 24, 0.5)), index_col=1)
            csv_data = csv_data.drop(csv_data.index[0])
            df = csv_data.melt(id_vars='CustomerID', var_name='Timestamp', value_name='Energy')
            reduced_df = pd.DataFrame(columns=['CustomerID', 'Timestamp', 'Energy'])

            # check if the required customer ids are contained in this data base, if not raise an error
            csv_customer_ids =  csv_data.CustomerID.unique()
            if not all([cid in csv_customer_ids for cid in customer_ids]):
                raise ValueError(f"Customer ID(s) {str(customer_ids)} could not be found in {path}")

            timestamps = df.Timestamp.unique()
            timestamp_origin = pd.Timestamp(datetime.now())
            timestamp_origin = timestamp_origin.floor(freq='D')

            for cid in customer_ids:
                cid_arr = (df.CustomerID == cid)
                for ts in timestamps:
                    bool_array = cid_arr & (df.Timestamp == ts)

                    df.loc[bool_array, 'Timestamp'] = \
                        pd.date_range(start=timestamp_origin + pd.Timedelta(hours=ts),
                                      freq='D', periods=np.count_nonzero(bool_array))

                    reduced_df = reduced_df.append(df.loc[bool_array])

            reduced_df['Month'] = reduced_df.Timestamp.dt.month
            reduced_df['DayOfWeek'] = reduced_df.Timestamp.dt.dayofweek
            reduced_df['Hour'] = reduced_df.Timestamp.dt.hour
            reduced_df.loc[reduced_df.Timestamp.dt.minute == 30, 'Hour'] += 0.5
            reduced_df.sort_values(by='Timestamp', inplace=True)
            reduced_df.to_csv('pivoted_loaddata.csv', index=False)

        self.data = reduced_df
