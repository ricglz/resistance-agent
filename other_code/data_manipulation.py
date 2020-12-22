#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main module"""

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    for filename in ['LoggerBot.log', 'vote_info_file.csv']:
        orig_filename = f'orig_{filename}'
        dataframe = read_csv(orig_filename, header=None)
        dataframe = dataframe.drop_duplicates()
        dataframe.to_csv(orig_filename, header=False, index=False)
        dataframe = DataFrame(MinMaxScaler().fit_transform(dataframe))
        dataframe.to_csv(filename, header=False, index=False)
