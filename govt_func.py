import pandas as pd
import datetime
from imp import reload
import math
import numpy as np
from dateutil.relativedelta import *
import statsmodels.api as sm


def get_quarter(df: pd.DataFrame):
    # extract Q1 and never change
    df_q1 = df.filter(like="Q1", axis=1)

    # if Q1 is nan, q2 = q2/2
    df_q2 = df.loc[
        :, lambda df: (df.columns.str[4:] == "Q1") | (df.columns.str[4:] == "Q2")
    ]

    for i in range(12):  # add an column that are all 0 at the beginning of each year
        year = 2011 + i
        df_q2.insert(i, str(year) + "Q0", 0)

    df_q2 = df_q2.sort_index(axis=1)
    # fjf;lakjflksaejflksajddi
    df_q2 = df_q2.interpolate(axis=1)  # if Q1 is nan, it would become q2/2
    df_q2 = df_q2.diff(1, axis=1)  # difference
    df_q2 = df_q2.filter(like="Q2", axis=1).replace(0, np.nan)

    # Q3
    df_q3 = df.loc[
        :, lambda df: (df.columns.str[4:] == "Q3") | (df.columns.str[4:] == "Q2")
    ]
    df_q3 = df_q3.diff(1, axis=1)
    df_q3 = df_q3.filter(like="Q3", axis=1).replace(0, np.nan)
    
    return pd.concat([df_q1, df_q2, df_q3], axis=1).sort_index(axis=1)
