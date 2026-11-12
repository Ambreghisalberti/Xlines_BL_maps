"""Functions copied verbatim from swapp/catalogues/__init__.py (swapp commit 23c5106)
(only the functions used by the pipeline, plus the functions they call)."""
import tscat
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import operator


TIME_DELTA_ATTR_MAP = (
        ('days', 'D'),
        ('seconds', 's'),
        ('microseconds', 'us')
        )


def to_timedelta64(value: timedelta) -> np.timedelta64:
    if isinstance(value, np.timedelta64):
        return value
    else:
        return reduce(operator.add,
            (np.timedelta64(getattr(value, attr), code)
            for attr, code in TIME_DELTA_ATTR_MAP if getattr(value, attr) > 0))


def read_catalogue_events(path):
    with tscat.Session():
        with open(path) as json_data:
            d = json.load(json_data)
            catalogue = d['events']
    return catalogue


def resolution_to_string(resolution):
    frequency = str(to_timedelta64(resolution))  # Ex: numpy.timedelta64(90,'s')
    [nbr, text] = frequency.split(' ')  # Ex: ['90','seconds']
    if text == 'nanoseconds':
        text = 'N'
    elif text == 'microseconds':
        text = 'U'
    elif text == 'milliseconds':
        text = 'L'
    elif text == 'seconds':
        text = 's'
    elif text == 'minutes':
        text = 'T'
    elif text == 'hours':
        text = 'H'
    elif text == 'days':
        text = 'D'

    else:
        raise Exception('The resolution is not in a format given explicitely in resolution_to_string. '
                        'The text form of this resolution must be explicited added in the function '
                        'resolution_to_string.')
    return nbr + text  # Ex: '90s'


def catalogue_to_edges_df(path,
                          resolution: np.timedelta64):
    """ Precondition: resolution has to be a string in the format '5S' for 5 seconds for example.
    It will round up dates to the closest 5s"""

    resolution = resolution_to_string(resolution)
    catalogue = read_catalogue_events(path)
    events = pd.DataFrame(columns=['begin', 'end'])
    for ev in catalogue:
        pd.to_datetime(catalogue[0]['start']).round(resolution)
        events.loc[len(events)] = {'begin': pd.to_datetime(ev['start']).round(resolution),
                                   'end': pd.to_datetime(ev['stop']).round(resolution)}

        # events.loc[len(events)] = {'begin':ev['start'][:19], 'end':ev['stop'][:19]}
        '''The [:19] is to have dates in seconds and not in between seconds 
        (for a problem of number of points in make_windows, of which I'm not sure about anymore)'''

    return events


