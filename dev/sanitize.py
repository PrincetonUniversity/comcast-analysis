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

CONVERT_OCTETS = 8 / (15 * 60 * 1024)

def aggregate_sum_data(name)
    """
    input: String name, output: ComcastDataSet(name)
    load a split by direction complete dataset
    aggregate service class names
    add datetime, throughput, IP columns
    and save it in filtered path.
    """
    # load agg-filtered dataset
    # already split by direction
    # ../filtered/ <datasetname>_up.pkl
    CDS = ComcastDataSet(FILENAME=name, **CONST)
    #CDF._get_default_name_from_filename()
    logger.debug(name + " load")
    CDF.load()

    # aggregate by summing service_class_name (default)
    # logger.debug(name + " aggregate")
    # CDS.aggregate()

    # add datetime and avg throughput columns
    #CDS.up['datetime'] = pd.to_datetime(CDS.up.end_time)
    #CDS.dw['datetime'] = pd.to_datetime(CDS.dw.end_time)
    #CDS.up['throughput'] = CDS.up.octets_passed * (9 / (15*60 * 1024 * 1024) )    #Mbps
    #CDS.dw['throughput'] = CDS.dw.octets_passed * (9 / (15*60 * 1024 * 1024) )    #Mbps
    #CDS.up['ipaddr'

    # pickle summed file
    logger.debug(name + " pickle unsanitized")
    CDS.up.to_pickle(CONST['SANITIZEDPATH'] + CDS.name + '_up_unsanitized.pkl')
    CDS.dw.to_pickle(CONST['SANITIZEDPATH'] + CDS.name + '_dw_unsanitized.pkl')
    return CDS

def slice_by_availability(df, AVAIL_THRESHOLD):
    """
    input: pd.Dataframe(ComcastDataSet-up/dw)
    output: sliced dataframe by availability
    loads unsanitized aggregate direction dataframe
    pivots to Device_number, datetime range and resamples
    calculates availability per device (can also do availability per date)
    slices based on availability threshold
    pickles sanitized data with avail at least AVAIL THRESHOLD
    """
    return

##############################################################################
# Helper functions for slice by availability
# taken from availability.py

def _load_unsanitized_df(dfname, direction):
    try:
        df = pd.read_pickle(CONST['SANITIZEDPATH']+dfname+'_'+direction+'_unsanitized.pkl')
    except:
        logger.warning("no directional unsanitized data to load " + dfname)
        return pd.DataFrame({})
    df['datetime'] = pd.to_datetime(df.end_time)
    df['throughput'] = df.octets_passed * CONVERT_OCTETS )    #Kbps
    return df

##############################################################################

def slice_by_month(test, control):
    """
    input: test and control dataframes after availability sanitization
    slice test to date range of control
    pickle sets in sanitized/<month>/
    """
    return
