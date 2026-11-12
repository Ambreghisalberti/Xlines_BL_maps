"""Functions copied verbatim from swapp/dataframes/__init__.py (swapp commit 23c5106)
(only the functions used by the pipeline, plus the functions they call)."""
import pandas as pd
import numpy as np


def get_consecutive_interval(times, dt=np.timedelta64(5, 's')):
    split = str(dt).split(' ')
    if split[1] == 'minutes':
        dt = str(dt).split(' ')[0] + 'min'
    else:
        dt = str(dt).split(' ')[0] + (str(dt).split(' ')[1][0])
    times = np.sort(times)
    all_times = pd.date_range(times[0], times[-1], freq=dt)
    temp = pd.DataFrame(np.zeros(len(all_times)), index=all_times, columns=['is_present'])
    temp.loc[times, 'is_present'] = 1
    temp['change'] = [0] + list((temp.is_present.values[1:] != temp.is_present.values[:-1]).astype('int'))
    temp['group'] = temp.change.values.cumsum()
    temp = temp.drop(temp[temp.is_present.values == 0].index.values)

    starts = temp.groupby('group').apply(lambda x: x.index.values[0]).values
    stops = temp.groupby('group').apply(lambda x: x.index.values[-1]).values
    assert (starts > stops).sum() == 0
    return starts, stops


