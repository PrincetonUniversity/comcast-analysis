from __future__ import division
import pandas as pd
import numpy as np
import random, os
import datetime
import matplotlib
import pickle as pkl
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter
import itertools
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

NUMENTRIES = 4000    #the last time slot should have at least a 1000 entries
AVAIL_MIN = 0.8
OUTPUTPATH = '/data/users/sarthak/comcast-data/sanitized/'
INPUTPATH = '/data/users/sarthak/comcast-data/filtered/'
SEPARATEDPATH = '/data/users/sarthak/comcast-data/separated/'
DFLIST = ['250-test']
CONTROLLIST = [ 'control'+str(x) for x in [1,2,3,4,5,6,7,8] ]
DFLIST.extend( CONTROLLIST )


def load_df(dfname):

    try:
        df = pd.read_pickle(INPUTPATH + dfname + '.pkl')
    except:
        logger.warning("no data to load " + dfname)
        return pd.DataFrame({})
    df['datetime'] = pd.to_datetime(df.end_time)
    df['throughput'] = df.octets_passed * (8 / (15*60 * 1024 * 1024) )    #Mbps
    return df

def pivot_table(df):
    pivoted = df.pivot(index='datetime', columns='Device_number', values='throughput')
    # the first/last date shouldn't be the absolute last, at least have N entries
    # to be considered as the last date, to avoid the heavy tail of dates with
    # just a couple of entrieNUMENTRIES = min( 3000, 0.5* len( pivoted.columns ) )
    ENTRIES = min( NUMENTRIES, 0.5* len( pivoted.columns ) )
    atleast_pivoted = pivoted[ pivoted.count(axis=1) > ENTRIES ]
    ts = pd.date_range(atleast_pivoted.index[0], atleast_pivoted.index[-1], freq='15min')

    resampled = pd.DataFrame(pivoted, index=ts)
    #resampled = pd.DataFrame(atleast_pivoted, index=ts)
    return resampled

def calculate_availability(pivoted_table):
    availability = pivoted_table.apply(lambda x: x.count()/len(x), axis=0)
    availability2 = pivoted_table.apply(lambda x: x.count()/len(x), axis=1)
    return availability, availability2

def plot_availability(availability, availability2, dfname, PATH=OUTPUTPATH+'availability/'):

    if not os.path.exists(PATH):
        os,makedirs(PATH)

    # CDF avail by device
    x = np.sort( availability )
    #x = 1-x
    x = x[::-1]
    y = range( len(availability) )
    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(x, y, marker='.', ls='')
    ax1.grid(1)
    ax1.set_xlabel('availability > x')
    ax1.set_ylabel('number of devices')
    fig1.savefig(OUTPUTPATH + 'availability/' + dfname+'-availability-CDF')

    # timeseries avail vs date
    fig2, ax2 = plt.subplots(1,1)
    availability2.plot(ax=ax2)
    ax2.set_xlabel("DateTime")
    ax2.set_ylabel("Availability per Time-Slot")
    fig2.savefig(OUTPUTPATH + 'availability/' + dfname+'-availability-by-date')

    logger.debug("availability " + dfname)
    return

def slice_df(availability, df):
    selected_devices = (availability[availability >= AVAIL_MIN]).index
    logger.debug( "Number of devices selected = " + str(len(selected_devices)) )
    sliced = df[df['Device_number'].isin(selected_devices)]
    logger.debug( "slice the df: entries = "+str( len(sliced) ) )
    sliced = sliced.sort(['Device_number', 'datetime'], ascending=[1,1])
    logger.debug( "Date range = " + str(sliced['datetime'].min()) + " to "
                 + str(sliced['datetime'].max()) )
    return sliced

def slice_and_plot_avail(name, direction):
    dfname = name + '_' + direction
    df = load_df(dfname)

    logger.debug("FULL " + dfname +" " + str( len(df['Device_number'].unique()) )
    + " " + str(df['datetime'].min()) + " " +str( df['datetime'].max() ) )

    p = pivot_table(df)
    a1, a2 = calculate_availability(p)
    plot_availability(a1, a2, dfname)
    #availability[dfname] = a1

    final_df = slice_df(a1, df)

    logger.debug( "SLICED " + dfname +" " + str( len(final_df['Device_number'].unique()) )
    + " " + str(final_df['datetime'].min() ) + " " +str( final_df['datetime'].max() ) )

    final_df.to_pickle(OUTPUTPATH + dfname + '.pkl')
    p = pivot_table(final_df)
    a1, a2 = calculate_availability(p)
    plot_availability(a1, a2, dfname, OUTPUTPATH+'final_availability/')

    del df
    del final_df
    del a1
    del a2

    return

def mp_slice():
    jobs = []
    for name in DFLIST:
        for direction in ['up', 'dw']:
            p = mp.Process(target=slice_and_plot_avail, args=(name, direction, ))
            jobs.append(p)
            p.start()
    return

def main_slice():
    availability = {}
    for name in DFLIST:
        for direction in ['up', 'dw']:
            dfname = name + '_' + direction
            df = load_df(dfname)

            logger.debug("FULL " + dfname +" " + str( len(df['Device_number'].unique()) )
            + " " + str(df['datetime'].min()) + " " +str( df['datetime'].max() ) )

            p = pivot_table(df)
            a1, a2 = calculate_availability(p)
            #plot_availability(a1, a2, dfname)
            availability[dfname] = a1

            final_df = slice_df(a1, df)

            logger.debug( "SLICED " + dfname +" " + str( len(final_df['Device_number'].unique()) )
            + " " + str(final_df['datetime'].min() ) + " " +str( final_df['datetime'].max() ) )

            final_df.to_pickle(OUTPUTPATH + dfname + '.pkl')
            #p = pivot_table(final_df)
            #a1, a2 = calculate_availability(p)
            #plot_availability(a1, a2, dfname, OUTPUTPATH+'final_availability/')

            del df
            del final_df
            del a1
            del a2

    pkl.dump(availability, open(OUTPUTPATH + 'availability.pkl', 'w'))
    return availability

def test():
    df = load_df('test-data-set')
    return df

def main_separate():
    for direction in ["up", "dw"]:

        test = '250-test_'+direction+'.pkl'
        df_test = pd.read_pickle(INPUTPATH + test)
        massive = []    # to combine control sets

        for control in ['control'+str(x)+'_'+direction+'.pkl' for x in range(1,9)]:
            controlname = control.split('.')[0]         #name includes direction
            OUTPUTPATH = SEPARATEDPATH + controlname + "/"
            if not os.path.exists(OUTPUTPATH):
                os.makedirs(OUTPUTPATH)

            df_control = pd.read_pickle(INPUTPATH + control)
            start_date = df_control['datetime'].min()
            end_date = df_control['datetime'].max()

            split_test = df_test[ (df_test['datetime'] <= end_date) & (df_test['datetime'] >= start_date) ]
            df_control.to_pickle(OUTPUTPATH + "control.pkl")
            split_test.to_pickle(OUTPUTPATH + "test.pkl")
            logger.debug(" control= "+controlname+" start_date= "+str(start_date)+
                    " end_date= "+str(end_date)+
                    " devices-control = "+str( len(df_control['Device_number'].unique()) )+
                    " devices-test = "+str( len(split_test['Device_number'].unique()) ) )

            massive.append(df_control)
            del df_control
            del split_test

        OUTPUTPATH = SEPARATEDPATH + "full_"+direction+"/"
        if not os.path.exists(OUTPUTPATH):
            os.makedirs(OUTPUTPATH)
        pd.concat(massive).to_pickle(OUTPUTPATH + "control.pkl")
        del massive
        df_test.to_pickle(OUTPUTPATH + "test.pkl")
    return

def filter_by_date(df, start_date, end_date):
    return df[ (df['datetime'] < end_date) & (df['datetime'] >= start_date) ]

def main_split_massive_by_month():
    for direction in ['up', 'dw']:
        INPUTPATH = SEPARATEDPATH + "full_"+direction+"/"
        df_test = pd.read_pickle(INPUTPATH + "test.pkl")
        df_control = pd.read_pickle(INPUTPATH + "control.pkl")

        start_dates = {"Oct": datetime.datetime(2014, 10, 1),
                    "Nov": datetime.datetime(2014, 11, 1),
                    "Dec": datetime.datetime(2014, 12, 1)}
        end_dates = {"Oct": datetime.datetime(2014, 10, 31),
                    "Nov": datetime.datetime(2014, 11, 30),
                    "Dec": datetime.datetime(2014, 12, 31)}
        for month in ["Oct", "Nov", "Dec"]:
            OUTPUTPATH = SEPARATEDPATH + month + "_"+direction+"/"
            if not os.path.exists(OUTPUTPATH):
                os.makedirs(OUTPUTPATH)
            cont = filter_by_date( df_control, start_dates[month], end_dates[month] )
            cont.to_pickle(OUTPUTPATH + "control.pkl")
            del cont
            test = filter_by_date( df_test, start_dates[month], end_dates[month] )
            test.to_pickle(OUTPUTPATH + "test.pkl")
            del test
            logger.debug("slice "+month+"_"+direction)
    return


def getSortedCDF(data):
    """
    return x,y for a cdf given one dimensional unsorted data x
    """
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

if __name__ is '__main__':
    pass
    #main_slice()
    #main_separate()
    #main_split_massive_by_month()
    #main()
