from __future__ import division
from ComcastDataSet import ComcastDataSet, CONST
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

CONST["SANITIZEDPATH"] = "/data/users/sarthak/comcast-data/sanitized/"

TESTSET = "250-test.dat"
CONTROLSETS = ["control"+str(x)+".dat" for x in range(1,9)]

AVAIL = 0.8
CONVERT_OCTETS = 8 / (15 * 60 * 1024)       # to kbps

def main(name):
    """
    input: String name, output: ComcastDataSet(name)
    load a split by direction complete dataset
    aggregate service class names
    add datetime, throughput, IP columns
    and save it in filtered path.
    """

    # load agg-filtered dataset
    # already split by direction, throughput aggregated, datetime
    # ../filtered/ <datasetname>_up.pkl
    CDS = ComcastDataSet(FILENAME=name, **CONST)
    logger.debug(name + " load")
    CDS.load()

    logger.debug(name + " sanitize with availability = "+str(AVAIL))
    slice_by_availability(CDS.up, AVAIL)
    slice_by_availability(CDS.dw, AVAIL)

    # pickle summed file
    logger.debug(name + " pickle sanitized up and dw")
    CDS.up.to_pickle(CONST['SANITIZEDPATH'] + CDS.name + '_up.pkl')
    CDS.dw.to_pickle(CONST['SANITIZEDPATH'] + CDS.name + '_dw.pkl')
    return CDS

def slice_by_availability(df, availability):
    """
    input: pd.Dataframe(ComcastDataSet-up/dw)
    output: sliced dataframe by availability
    loads unsanitized aggregate direction dataframe
    pivots to Device_number, datetime range and resamples
    calculates availability per device (can also do availability per date)
    slices based on availability threshold
    pickles sanitized data with avail at least AVAIL THRESHOLD
    """
    logger.debug("pivot table")
    p = _pivot_table(df)
    return

##############################################################################
# Helper functions for slice by availability
# taken from availability.py
##############################################################################

def _load_unsanitized_df(dfname, direction):
    try:
        df = pd.read_pickle(CONST['SANITIZEDPATH']+dfname+'_'+direction+'_unsanitized.pkl')
    except:
        logger.warning("no directional unsanitized data to load " + dfname)
        return pd.DataFrame({})
    df['datetime'] = pd.to_datetime(df.end_time)
    df['throughput'] = df.octets_passed * CONVERT_OCTETS #Kbps
    return df

def _pivot_table(df):
    """
    make it DATETIME x MACHINES
    """

    pivoted = df.pivot(index='datetime', columns='Device_number', values='throughput')

    # select min of half the machines, or 3k machines
    NUMENTRIES = min( 5000, 0.5* len( pivoted.columns ) )

    # each time should have at least NUMENTRIES, filter out the rest of the
    # time stamps, create a proper time series range
    atleast_pivoted = pivoted[ pivoted.count(axis=1) > NUMENTRIES ]
    ts = pd.date_range(atleast_pivoted.index[0], atleast_pivoted.index[-1], freq='15min')

    # resample atleast_pivoted dataframe with completed time series, adding
    # missing timeslots
    resampled = pd.DataFrame(atleast_pivoted, index=ts)

    return resampled

##############################################################################

def slice_by_month(test, control):
    """
    input: test and control dataframes after availability sanitization
    slice test to date range of control
    pickle sets in sanitized/<month>/
    """
    return

def test():
    CDS = ComcastDataSet(FILENAME='test-data-set', **CONST)
    CDS.load()
    rs = _pivot_table(CDS.up)
    return CDS, rs
