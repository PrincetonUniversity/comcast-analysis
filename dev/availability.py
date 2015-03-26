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

NUMENTRIES = 1000    #the last time slot should have at least a 1000 entries
AVAIL_MIN = 0.8
SLICEDPATH = '../sliced/'
INPUTPATH = '../filtered/'
DFLIST = ['250-test']
CONTROLLIST = [ 'control'+str(x) for x in [1,2,3,4,5,6,7,8] ]
DFLIST.extend( CONTROLLIST )


def load_df(dfname):

    try:
        df = pd.read_pickle(INPUTPATH + dfname + '_dw_summed.pkl')
    except:
        logger.warning("no data to load " + dfname)
        return pd.DataFrame({})
    df['datetime'] = pd.to_datetime(df.end_time)
    df['throughput'] = df.octets_passed * (8 / (15*60 * 1024 * 1024) )    #Mbps
    return df

def pivot_table(df, dfname):
    pivoted = df.pivot(index='datetime', columns='Device_number', values='throughput')
    # the first/last date shouldn't be the absolute last, at least have N entries
    # to be considered as the last date, to avoid the heavy tail of dates with
    # just a couple of entries
    NUMENTRIES = min( 3000, 0.5* len( pivoted.columns ) )
    logger.debug("dfname = " + dfname + "; NUMENTRIES = "+str(NUMENTRIES))
    atleast_pivoted = pivoted[ pivoted.count(axis=1) > NUMENTRIES ]
    ts = pd.date_range(atleast_pivoted.index[0], atleast_pivoted.index[-1], freq='15min')

    resampled = pd.DataFrame(pivoted, index=ts)
    return resampled

def plot_availability(pivoted_table, dfname):
    availability = pivoted_table.apply(lambda x: x.count()/len(x), axis=0)
    x = np.sort( availability )
    #x = 1-x
    x = x[::-1]
    y = range( len(availability) )
    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(x, y, marker='.', ls='')
    ax1.grid(1)
    ax1.set_xlabel('availability > x')
    ax1.set_ylabel('number of devices')
    fig1.savefig(SLICEDPATH+dfname+'-availability')

    availability2 = pivoted_table.apply(lambda x: x.count()/len(x), axis=1)
    fig2, ax2 = plt.subplots(1,1)
    availability2.plot(ax=ax2)
    ax2.set_xlabel("DateTime")
    ax2.set_ylabel("Availability per Time-Slot")
    fig2.savefig(SLICEDPATH+dfname+'-availability-by-date')

    logger.debug("availability " + dfname)
    return availability

def slice_df(availability, df):
    selected_devices = (availability[availability >= AVAIL_MIN]).index
    logger.debug( "Number of devices selected = " + str(len(selected_devices)) )
    sliced = df[df['Device_number'].isin(selected_devices)]
    logger.debug( "slice the df: entries = "+str( len(sliced) ) )
    return sliced.sort(['Device_number', 'datetime'], ascending=[1,1])

def main_slice():
    availability = {}
    for dfname in DFLIST:

        df = load_df(dfname)
        p = pivot_table(df, dfname)
        avail = plot_availability(p, dfname)
        availability[dfname] = avail

        final_df = slice_df(avail, df)
        logger.debug( dfname +" " + str( len(final_df['Device_number'].unique()) )
        + " " + str(final_df.iloc[0]['datetime']) + " " +str( final_df.iloc[-1]['datetime'] ) )

        final_df.to_pickle(SLICEDPATH + dfname+'-sliced.pkl')

        del df
        del final_df
        del avail

    pkl.dump(availability, open(SLICEDPATH + 'availability.pkl', 'w'))
    return availability

def main_separate():
    test = '250-test-sliced.pkl'
    df_test = pd.read_pickle(SLICEDPATH + test)
    massive = []    # to combine control sets
    SEPARATEDPATH = "../separated/"

    for control in ['control'+str(x)+'-sliced.pkl' for x in range(1,9)]:
        controlname = control.split("-")[0]
        SEPARATEDPATH = "../separated/" + controlname + "/"
        if not os.path.exists(SEPARATEDPATH):
            os.makedirs(SEPARATEDPATH)

        df_control = pd.read_pickle(SLICEDPATH + control)
        start_date = df_control['datetime'].min()
        end_date = df_control['datetime'].max()

        split_test = df_test[ (df_test['datetime'] < end_date) & (df_test['datetime'] >= start_date) ]
        df_control.to_pickle(SEPARATEDPATH + controlname + ".pkl")
        split_test.to_pickle(SEPARATEDPATH + "test" + ".pkl")
        logger.debug(" control = "+controlname+" start_date = "+str(start_date)+
                   " end_date = "+str(end_date)+
                   " devices - control = "+str( len(df_control['Device_number'].unique()) )+
                   " devices - 250-test = "+str( len(split_test['Device_number'].unique()) ) )

        massive.append(df_control)
        del df_control
        del split_test

    SEPARATEDPATH = "../separated/full/"
    if not os.path.exists(SEPARATEDPATH):
        os.makedirs(SEPARATEDPATH)
    pd.concat(massive).to_pickle(SEPARATEDPATH + "control.pkl")
    del massive
    df_test.to_pickle(SEPARATEDPATH + "test.pkl")
    return

def getSortedCDF(data):
    """
    return x,y for a cdf given one dimensional unsorted data x
    """
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

def main_cdf():

    fig1, ax1 = plt.subplots(1,1)
    marker = itertools.cycle((',', '+', '1', '.', 'o', '*', '2', 's', 'p', 'd'))
    ls = itertools.cycle(('-','--',':'))

    for dfname in DFLIST:
        df = pd.read_pickle(SLICEDPATH + dfname+'-sliced.pkl')
        series = list(df['throughput'])
        calcMean = '%.2E' % np.mean( series )
        logger.debug( "plot CDF "+dfname+" length "+str(len(series))+ " mean "+str(calcMean) )
        x, y = getSortedCDF( series )
        ax1.plot(x, y, label=dfname+': '+str(calcMean), linestyle=ls.next(), marker=marker.next(), markevery=len(x)/10)
    ax1.set_xscale('log')
    ax1.set_xlabel('Throughput [Mbps]')
    ax1.set_ylabel('CDF')
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(SLICEDPATH + 'cdf/' + 'CDF-throughput-all')
    return

def combo_cdf():

    fig1, ax1 = plt.subplots(1,1)
    #marker = itertools.cycle((',', '+', '1', '.', 'o', '*', '2', 's', 'p', 'd'))
    #ls = itertools.cycle(('-','--',':'))

    dfname = '250-test'
    df = pd.read_pickle(SLICEDPATH + dfname+'-sliced.pkl')
    series = list(df['throughput'])
    calcMean = '%.2E' % np.mean( series )
    logger.debug( "plot CDF "+dfname+" length "+str(len(series))+ " mean "+str(calcMean) )
    x, y = getSortedCDF( series )
    ax1.plot(x, y, label=dfname+': '+str(calcMean), marker='o',
             markevery=len(y)/10, markeredgecolor=None)

    control = []
    for dfname in CONTROLLIST:
        df = pd.read_pickle(SLICEDPATH + dfname+'-sliced.pkl')
        series = list(df['throughput'])
        calcMean = '%.2E' % np.mean( series )
        logger.debug( "load "+dfname+" length "+str(len(series))+ " mean "+str(calcMean) )
        control.extend( series )

    dfname = 'control'
    calcMean = '%.2E' % np.mean( control )
    x, y = getSortedCDF( control )
    logger.debug( "plot CDF "+dfname+" length "+str(len(control))+ " mean "+str(calcMean) )
    ax1.plot(x, y, label=dfname+': '+str(calcMean), marker='d',
             markevery=len(y)/10, markeredgecolor=None)
    ax1.set_xscale('log')
    ax1.set_xlabel('Throughput [Mbps]')
    ax1.set_ylabel('CDF')
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(SLICEDPATH + 'cdf/' + 'CDF-throughput-combo')
    return


def main_scatter_all():

    for dfname in DFLIST:
        fig1, ax1 = plt.subplots(1,1, figsize=(20,10))
        df = pd.read_pickle(SLICEDPATH + dfname+'-sliced.pkl')
        logger.debug( "plot scatter "+dfname )
        ax1.scatter(list( df['datetime'] ), list( df['throughput'] ),
                    marker='.', edgecolors='none')
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Throughput [Mbps]')
        x0 = datetime.datetime(2014, 9, 30)
        x1 = datetime.datetime(2014, 12, 30)
        ax1.set_xlim([x0, x1])
        ax1.set_ylim([-10, 350])
        ax1.set_title(dfname)
        ax1.grid(1)
        #ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(SLICEDPATH + 'timeseries/' + 'all-throughput-'+dfname)
    return


if __name__ is '__main__':
    #main_slice()
    main_separate()
    # main_cdf()
    #combo_cdf()
    #main_scatter_all()
    #main()
