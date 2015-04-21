from __future__ import division
import pickle as pkl
import pandas as pd
import numpy as np
import os, sys
from collections import defaultdict
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

CONVERT_OCTETS = 8 / (15 * 60 * 1024)    #BYTES TO kbps

# CDF
def getSortedCDF(data):
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

# Peak Ratio
def throughput_stats_per_device_per_date(dfin):
    """
    gets throughput description over each day for each device
    groupby device, date
    Input: pandas dataframe; Columns = [Device_number, datetime, throughput]
    Output: pandas dataframe; Index = [Device_number, date]; Columns=[perc90, median, sum, len, std,...]
    """

    # make a copy and for each device get stats on each day
    df = dfin.copy()
    g = df.set_index('datetime').groupby(["Device_number"])
    throughput_stats = g['throughput'].resample('D', how=[np.sum, len, max, min, np.median, np.mean, np.std])

    # for perc90 need to do it separately
    df['date'] = df['datetime'].apply(lambda x: x.date())
    # df['time'] = df['datetime'].apply(lambda x: x.time())
    g = df.groupby(['Device_number', 'date'], as_index=False)
    perc90_per_day = g['throughput'].apply(lambda x: np.percentile(x, 95))

    throughput_stats['perc90'] = perc90_per_day
    throughput_stats = throughput_stats.rename(columns={'datetime':'date'})

    return throughput_stats

def aggregate_octets_stats_per_datetime(df):
    """
    aggregate bytes across devices for each datetime
    return pandas dataframe; Index=datetime; Columns=[day, time, sum, len, perc90, median, max, min, std, ...]
    """

    # when applied on df.agg() this field gets renamed to '<lambda>'
    perc90 = lambda x: x.quantile(0.95)

    # dataframe indexed by datetime with aggregate columns
    g = df.groupby('datetime')
    agg_octets = g['octets_passed'].agg([np.sum, np.mean, np.median, perc90, max, min, np.std]).rename(columns={'<lambda>':'perc90'})

    # add corresponding timestamp and weekday to columns
    agg_octets['time'] = agg_octets.index.time
    agg_octets['day'] = agg_octets.index.weekday

    return agg_octets

'''
# Prime-Time Ratio
def primetime_stats_per_device_per_date(dfin):
    """
    gets hourly bytes passed for each device each day
    groupby device, date, hour
    Input: pandas dataframe; Columns = [Device_number, datetime, octets_passed]
    Output: pandas dataframe; Index = [Device_number, date; hour]; Columns=[octets_passed] (total)
    """

    # make a copy and for each device get stats on each day
    df = dfin.copy()
    g = df.set_index('datetime').groupby(["Device_number"])
    primetime_stats = g['throughput'].resample('1H', how=[np.sum])

    # for perc90 need to do it separately
    df['date'] = df['datetime'].apply(lambda x: x.date())
    # df['time'] = df['datetime'].apply(lambda x: x.time())
    g = df.groupby(['Device_number', 'date'], as_index=False)
    perc90_per_day = g['throughput'].apply(lambda x: np.percentile(x, 90))

    throughput_stats['perc90'] = perc90_per_day

    return throughput_stats
'''

def ratios_per_date(peak_ratio, param='mean'):
    """
    mean over devices grouped by date: variation of mean PT ratio with time
    average PT ratio of all devices over one day
    Input: pandas dataframe [Device_number, date, peakratio]
    Output: pandas dataframe Index = date; Columns = 0 (peakratio aggregated)
    COMMENT: take max(), min(), median(), perc90() over devices to separate outliers
    """

    # peak_ratio['peakratio'] is 90perc:median per day(datetime) per device
    g = peak_ratio.groupby('date')['peakratio']

    # basically return g.<param>()
    if param in ['mean', 'sum', 'median', 'max', 'min']:
        ratios = getattr(g, param)()
    elif param == 'perc90':
        #ratios = g.quantile(0.95)
        ratios = getattr(g, 'quantile')(0.95)
    else:
        logger.warning("unknown parameter to aggregate peak-ratio per date")
        return None
    return ratios

def ratios_per_device(peak_ratio, param='mean'):
    """
    mean over dates grouped by device: variation of mean PT ratio with device
    average PT ratio of a device over its lifetime
    Input: pandas dataframe [Device_number, date, peakratio]
    Output: pandas dataframe Index = Device_number; Columns = 0 (peakratio aggregated)
    separates outliers (business vs aggressive) by mean PT ratio
    COMMENT: maybe should be applied only on per day basis or WEEKDAYS only
    """

    # peak_ratio['peakratio'] is 90perc:median per day(datetime) per device
    g = peak_ratio.groupby('Device_number')['peakratio']

    # basically return g.<param>()
    if param in ['mean', 'sum', 'median', 'max', 'min']:
        ratios = getattr(g, param)()
    elif param == 'perc90':
        #ratios = g.quantile(0.95)
        ratios = getattr(g, 'quantile')(0.95)
    else:
        logger.warning("unknown parameter to aggregate peak-ratio per device")
        return None
    return ratios

def get_peak_ratios(throughput_stats, numer='perc90', denom='median'):
    """
    all ratios regardlesss of device/date
    Input: pandas dataframe of throughput_stats; Index = [Device_number, date]; Columns = [sum, len, max, min ...]
    Output: pandas dataframe of ratio of throughput_stats columns. Reindexed. Columns = [Device_number. date, peakratio]
    Default: perc90 (numer): median (denom) per day per device
    """
    ratios = (throughput_stats[numer]/throughput_stats[denom]).reset_index().rename(columns={'datetime':'date', 0:'peakratio'})
    return ratios

# Plotter - timeseries
def plot_initial_timeseries(g1, g2, param, PLOTPATH):
    """
    plot timeseries of data rate (mean, max, median of devices) per datetime
    Input: pandas grouped by datetime; Index = datetime; Columns = throughput (over all devices)
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))

    # get the right timeseries
    if param in ['sum', 'max', 'min', 'median', 'mean']:
        ts1 = getattr(g1['throughput'], param)()
        ts2 = getattr(g2['throughput'], param)()
    elif param == 'perc90':
        ts1 = getattr(g1['throughput'], 'quantile')(0.95)
        ts2 = getattr(g2['throughput'], 'quantile')(0.95)
    else:
        logger.warning("unknown parameter to plot throughput timeseries")
        plt.close()
        return

    # plot the time series
    ts1.plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, label='250')
    ts2.plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, label='105')

    # save with a filename containing the aggregation parameter
    filename_label = param.upper()
    ax1.set_ylabel('Data Rate [kbps]')
    ax1.set_title(param+' Avg Data Rate')

    plotname = 'timeseries-throughput-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_peak_ratio_timeseries(rperday1, rperday2, agg_param, PLOTPATH):
    """
    plot timeseries of peak-ratio (mean, max, median of devices) per day
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))

    # x-axis: DAY, y-axis: agg (param) ratio over devices
    rperday1.plot(ax=ax1, marker='o', linestyle='--', label='250')
    rperday2.plot(ax=ax1, marker='d', linestyle='--', label='105')

    filename_label = agg_param.upper()
    ax1.set_ylabel(agg_param+' peak-ratio')
    ax1.set_yscale('log')
    ax1.set_title("Daily peak-ratio (95%:average) aggregated "+agg_param+" over devices")

    plotname = 'peakratio-timeseries-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_peak_ratio_cdf(rperdev1, rperdev2, agg_param, PLOTPATH):
    """
    plot CDF of peak-ratio all, or peak-ratio per device
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))

    # x-axis: peak-ratio per device, y-axis: agg (param) ratio over days
    x1,y1 = getSortedCDF(rperdev1.values)
    x2,y2 = getSortedCDF(rperdev2.values)
    ax1.plot(x1, y1, marker='o', markevery=len(y1)//10, linestyle='--', label='250')
    ax1.plot(x2, y2, marker='d', markevery=len(y2)//10, linestyle='--', label='105')

    filename_label = agg_param.upper()
    ax1.set_xlabel(agg_param + ' peak-ratio per device')
    ax1.set_xscale('log')
    ax1.set_title("Distribution of peak-ratio (95%:average) per device aggregated "+agg_param+" over days")

    plotname = 'peakratio-CDF-devices-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_octets_per_day(g1, g2, param_device, param_time, PLOTPATH):
    """
    Input: pandas groupedby; Index=['time', 'day']; Columns=[sum, perc90, max, min, median, ...]
    param_device: denotes the agg of bytes in a particular datetime over devices for the timeseries
    param_time: denotes the agg over [week, time] group: median, mean, perc90, max
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,8))
    tit=param_time

    if param_time in ['mean', 'max', 'min', 'median']:
        ts1 = getattr(g1[param_device], param_time)
        ts2 = getattr(g2[param_device], param_time)

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='250')
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='105')

    elif param_time == 'perc90':
        ts1 = g1[param_device].quantile(0.95)
        ts2 = g2[param_device],quantile(0.95)

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='250')
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='105')

    elif param_time == 'all1':
        logger.debug("plot 90-%ile, median")

        g1[param_device].quantile(0.95).plot(ax=ax1, color='b', linestyle='-', label='250')
        g2[param_device].quantile(0.95).plot(ax=ax1, color='g', linestyle='-', label='105')

        g1[param_device].median().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        g2[param_device].median().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

        tit = 'perc95, median'

    elif param_time == 'all2':
        logger.debug("plot max, mean")

        g1[param_device].max().plot(ax=ax1, color='b', linestyle='-', label='250')
        g2[param_device].max().plot(ax=ax1, color='g', linestyle='-', label='105')

        g1[param_device].mean().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        g2[param_device].mean().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

        tit = 'max, mean'

    else:
        logger.debug("no param_time selected so plot 90-%ile, median, max, mean")

        g1[param_device].max().plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='250')
        g2[param_device].max().plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='105')

        g1[param_device].quantile(0.95).plot(ax=ax1, color='k', linestyle='-', label='250')
        g2[param_device].quantile(0.95).plot(ax=ax1, color='r', linestyle='-', label='105')

        g1[param_device].mean().plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='250')
        g2[param_device].mean().plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='105')

        g1[param_device].median().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        g2[param_device].median().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

    ax1.set_ylabel(param_time + '$_{time}$ '+param_device+'$_{device}$ Bytes')
    ax1.set_title(tit+" Bytes in a 15 min slot")

    filename_label = param_time.upper()
    plotname = 'describe-total-octets-per-day-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_throughput_per_day(g1, g2, param_device, param_time, PLOTPATH):
    """
    Same as above, just convert octets to throughput
    Input: pandas groupedby; Index=['time', 'day']; Columns=[sum, perc90, max, min, median, ...]
    param_device: denotes the agg of bytes in a particular datetime over devices for the timeseries
    param_time: denotes the agg over [week, time] group: median, mean, perc90, max
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,8))
    tit=param_time

    if param_time in ['mean', 'max', 'min', 'median']:
        ts1 = getattr(g1[param_device], param_time) * CONVERT_OCTETS
        ts2 = getattr(g2[param_device], param_time) * CONVERT_OCTETS

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-'+param_time)
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-'+param_time)

    elif param_time == 'perc90':
        ts1 = g1[param_device].quantile(0.95)* CONVERT_OCTETS
        ts2 = g2[param_device],quantile(0.95)* CONVERT_OCTETS

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-'+param_time)
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-'+param_time)

    elif param_time == 'all1':
        logger.debug("plot perc90, median")
        tit = 'perc95, median'

        (g1[param_device].quantile(0.95)* CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle='-', label='250')
        (g2[param_device].quantile(0.95)* CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle='-', label='105')

        (g1[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        (g2[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

    elif param_time == 'all2':
        logger.debug("plot max, mean")
        tit = 'max, mean'

        (g1[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle='-', label='250')
        (g2[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle='-', label='105')

        (g1[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        (g2[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

    else:
        logger.debug("no param_time selected so plot 90-%ile, median, max, mean over time")

        (g1[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='250')
        (g2[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='105')

        (g1[param_device].quantile(0.95)* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='-', label='250')
        (g2[param_device].quantile(0.95)* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='-', label='105')

        (g1[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='250')
        (g2[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='105')

        (g1[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250')
        (g2[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='105')

    ax1.set_ylabel(param_time + '$_{time}$ '+param_device+'$_{device}$ Data Rate [kbps]')
    ax1.set_title(tit + " Data Rate in a 15 min slot")

    filename_label = param_time.upper()
    plotname = 'describe-total-throughput-per-day-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def _get_peak_primetime_per_set(test_full, control_full, PLOTPATH):
    """
    For each 4-hour period in a day:
        Splits df into peak-time and non-peak time
        calculates the peak-time load as mean of throughput OR sum of octets

    """
    # Add time column

    if 'time' not in test_full.columns:
        test_full['time'] = test_full.set_index('datetime').index.time
    if 'time' not in control_full.columns:
        control_full['time'] = control_full.set_index('datetime').index.time

    if (os.path.isfile(PLOTPATH + "df_best_primetime_hour.pkl")):
        logger.info("Load df_best_primetime_hour.pkl")
        return pd.read_pickle(PLOTPATH + "df_best_primetime_hour.pkl")

    HRS_IN_DAY = 24
    PEAK_HRS = 4
    NONPEAK_HRS = HRS_IN_DAY - PEAK_HRS
    # get avg peak vs non peak throughput per day

    logger.debug("Calculate df_primetime for each "+str(PEAK_HRS)+" slot in a day")

    myData = defaultdict(list)

    for ctr in range(HRS_IN_DAY):
        start_time = datetime.time(ctr,0)
        stop_time = datetime.time( (ctr+ PEAK_HRS) % HRS_IN_DAY )

        df = test_full
        if stop_time > start_time:
            peak_t = df[ (df['time'] < stop_time) & (df['time'] >= start_time) ]
            nonpeak_t = df[ (df['time'] > stop_time) | (df['time'] <= start_time) ]
        else:
            nonpeak_t = df[ (df['time'] > stop_time) & (df['time'] <= start_time) ]
            peak_t = df[ (df['time'] < stop_time) | (df['time'] >= start_time) ]

        df = control_full
        if stop_time > start_time:
            peak_c = df[ (df['time'] < stop_time) & (df['time'] >= start_time) ]
            nonpeak_c = df[ (df['time'] > stop_time) | (df['time'] <= start_time) ]
        else:
            nonpeak_c = df[ (df['time'] > stop_time) & (df['time'] <= start_time) ]
            peak_c = df[ (df['time'] < stop_time) | (df['time'] >= start_time) ]

        '''
        # to calculate prime time ratio per day for each hour
        if how = 'mean':
            test_peak = peak_t.groupby('date')['throughput'].mean()
            test_nonpeak = nonpeak_t.groupby('date')['throughput'].mean()

            control_peak = peak_c.groupby('date')['throughput'].mean()
            control_nonpeak = nonpeak_c.groupby('date')['throughput'].mean()
        elif how='sum':
            test_peak = peak_t.groupby('date')['octets_passed'].sum()
            test_nonpeak = nonpeak_t.groupby('date')['octets_passed'].sum()

            control_peak = peak_c.groupby('date')['octets_passed'].sum()
            control_nonpeak = nonpeak_c.groupby('date')['octets_passed'].sum()
        '''

        myData['start_time'].append( start_time )
        myData['stop_time'].append( stop_time )

        myData['peak_t'].append( peak_t['throughput'].mean() )
        myData['nonpeak_t'].append( nonpeak_t['throughput'].mean() )
        myData['peak_c'].append( peak_c['throughput'].mean() )
        myData['nonpeak_c'].append( nonpeak_c['throughput'].mean() )
        myData['peak_t data'].append( peak_t['octets_passed'].sum() )
        myData['nonpeak_t data'].append( nonpeak_t['octets_passed'].sum() )
        myData['peak_c data'].append( peak_c['octets_passed'].sum() )
        myData['nonpeak_c data'].append( nonpeak_c['octets_passed'].sum() )

    df_primetime = pd.DataFrame(myData)
    df_primetime['test_ratio [rate]'] = df_primetime['peak_t']/df_primetime['nonpeak_t']
    df_primetime['control_ratio [rate]'] = df_primetime['peak_c']/df_primetime['nonpeak_c']
    df_primetime['test_ratio [data]'] = df_primetime['peak_t data']/df_primetime['nonpeak_t data']
    df_primetime['control_ratio [data]'] = df_primetime['peak_c data']/df_primetime['nonpeak_c data']
    df_primetime['test_ratio'] = (df_primetime['peak_t data']/PEAK_HRS) / (df_primetime['nonpeak_t data']/NONPEAK_HRS)
    df_primetime['control_ratio'] = (df_primetime['peak_c data']/PEAK_HRS) / (df_primetime['nonpeak_c data']/NONPEAK_HRS)

    df_primetime.to_pickle(PLOTPATH + "df_best_primetime_hour.pkl")
    df_primetime.to_html(PLOTPATH + "df_best_primetime_hour.html")
    return df_primetime

def get_peak_nonpeak_series(test_full, control_full, PLOTPATH):

    # peak prime time at each hour in a day, calculated with throughput or data
    # this also adds the time column
    df_primetime = _get_peak_primetime_per_set(test_full, control_full, PLOTPATH)

    # find the best MAX primetime hours
    start_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['start_time']
    stop_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("test set " + str(start_time_t) +" "+ str(stop_time_t))
    start_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['start_time']
    stop_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("control set " + str(start_time_c) +" "+ str(stop_time_c))
    del df_primetime

    df = test_full # Salt Lake City on PST => 7-11pm == 2-6 AM
    if stop_time_t < start_time_t:
        bool_series = (df['time'] < stop_time_t) | (df['time'] >= start_time_t)
    else:
        bool_series = (df['time'] < stop_time_t) & (df['time'] >= start_time_t)
    peak_t = df[ bool_series ]
    nonpeak_t = df[ ~ bool_series ]

    df = control_full # Still SLC
    if stop_time_c < start_time_c:
        bool_series = (df['time'] < stop_time_c) | (df['time'] >= start_time_c)
    else:
        bool_series = (df['time'] < stop_time_c) & (df['time'] >= start_time_c)
    peak_c = df[ bool_series ]
    nonpeak_c = df[ ~ bool_series ]

    return peak_t, nonpeak_t, peak_c, nonpeak_c

def _avg_bytes_per_hr(df):
    """
    Input: pandas df peak_t, nonpeak_t, peak_c, nonpeak_c
    Output: pandas df [ Device_number | date | bytes_per_hr ] where bph is avg
    in an hour during that day
    TODO: make the input test_full instead
    """
    # sum of bytes per device per hour: resample by hour and for each device
    # group, sum the octets_passed in that hour. drop na entries later for
    # samples outside peak times that got resampled
    g = df.set_index('datetime').groupby("Device_number")
    summed_per_hr = g['octets_passed'].resample('H', how=sum).dropna()

    # stick to the definition: take avg bytes send in a peak hour, per device per day
    avg_per_day = summed_per_hr.reset_index().set_index("datetime").groupby(
        "Device_number").resample("D", how=np.mean).reset_index().rename(
            columns={0:'bytes_per_hr', 'datetime':'date'})
    return avg_per_day.pivot(index='date', columns='Device_number', values='bytes_per_hr')

def get_primetime_ratio(peak_t, nonpeak_t, peak_c, nonpeak_c):
    """
    peak, nonpeak are original dataframes sliced based on prime time hours
    - field should be (1) octets_passed, in wchich case we will sum all octets
    per hour per device, then get average (or max/90perc) octets per hour for
    that device, and then take 'param' over devices, which will be mean, median
    , etc. (2) throughput, in which case we will first get avg (or max/90perc)
    throughput in peak_time per day per device, then take 'param' over devices
    """
    # get ratio (peak:nonpeak avg bytes per hour) for each (device, date)
    peak_t = _avg_bytes_per_hr(peak_t)
    nonpeak_t = _avg_bytes_per_hr(nonpeak_t)
    peak_c = _avg_bytes_per_hr(peak_c)
    nonpeak_c = _avg_bytes_per_hr(nonpeak_c)
    r_test = (peak_t/nonpeak_t).stack().reset_index().rename(columns={0:'ratio'})
    r_control = (peak_c/nonpeak_c).stack().reset_index().rename(columns={0:'ratio'})

    # now we can get mean, max, perc90 peak primetime per day
    #r_test = getattr(peak_t.groupby('date')[field], param)() / getattr(nonpeak_t.groupby('date')[field], param)()
    #r_control = getattr(peak_c.groupby('date')[field], param)() / getattr(nonpeak_c.groupby('date')[field], param)()
    return r_test, r_control

def plot_primetime_ratio_by_date(r_test, r_control, param, PLOTPATH):
    # Timeseries
    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))
    tit = param

    r_test_g = r_test.groupby('date')['ratio']
    r_control_g = r_control.groupby('date')['ratio']

    if param in ['mean', 'median', 'max']:
        r_t = getattr(r_test_g, param)()
        r_c = getattr(r_control_g, param)()

    elif param=='perc90':
        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)

    elif param=='all1':
        logger.debug("Plot: median, perc90")
        r_t2 = getattr(r_test_g, 'median')()
        r_c2 = getattr(r_control_g, 'median')()
        r_t2.plot(ax=ax1, marker='o', color='k', linestyle='--', label='250')
        r_c2.plot(ax=ax1, marker='d', color='r', linestyle='--', label='105')

        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)
        tit = 'median, perc95'

    elif param=='all2':
        logger.debug("Plot: mean, max")
        r_t2 = getattr(r_test_g, 'mean')()
        r_c2 = getattr(r_control_g, 'mean')()
        r_t2.plot(ax=ax1, marker='o', color='k', linestyle='--', label='250')
        r_c2.plot(ax=ax1, marker='d', color='r', linestyle='--', label='105')

        r_t = getattr(r_test_g, 'max')()
        r_c = getattr(r_control_g, 'max')()
        tit = 'mean, max'

    else:
        logger.debug("Plot all: median, perc90")
        r_t2 = getattr(r_test_g, 'median')()
        r_c2 = getattr(r_control_g, 'median')()
        r_t2.plot(ax=ax1, marker='o', color='k', linestyle='--', label='250')
        r_c2.plot(ax=ax1, marker='d', color='r', linestyle='--', label='105')

        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)

    r_t.plot(ax=ax1, color='b', marker='o', label='250')
    r_c.plot(ax=ax1, color='g', marker='d', label='105')

    ax1.set_ylabel('Prime-time ratio')
    #ax1.set_yscale('log')
    ax1.set_title("Prime-time Ratio every Date - "+tit)

    filename_label=param.upper()
    plotname = 'prime-time-ratio-by-date-timeseries-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_primetime_ratio_per_device(r_test, r_control, param, PLOTPATH):
    # CDF
    fig1, ax1 = plt.subplots(1,1)
    r_test_g = r_test.groupby('Device_number')['ratio']
    r_control_g = r_control.groupby('Device_number')['ratio']
    tit = param

    if param in ['mean', 'median', 'max']:
        r_t = getattr(r_test_g, param)()
        r_c = getattr(r_control_g, param)()
    elif param=='perc90':
        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)

    elif param=='all1':
        logger.debug("Plot: median, perc90")
        r_t2 = getattr(r_test_g, 'median')()
        r_c2 = getattr(r_control_g, 'median')()
        x,y = getSortedCDF(r_t2)
        ax1.plot(x, y, marker='o', color='k', linestyle='--', label='250', markevery=len(y)//10)
        x,y = getSortedCDF(r_c2)
        ax1.plot(x, y, marker='d', color='r', linestyle='--', label='105', markevery=len(y)//10)

        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)
        tit = 'median, perc95'

    elif param=='all2':
        logger.debug("Plot: mean, max")
        r_t2 = getattr(r_test_g, 'mean')()
        r_c2 = getattr(r_control_g, 'mean')()
        x,y = getSortedCDF(r_t2)
        ax1.plot(x, y, marker='o', color='k', linestyle='--', label='250', markevery=len(y)//10)
        x,y = getSortedCDF(r_c2)
        ax1.plot(x, y, marker='d', color='r', linestyle='--', label='105', markevery=len(y)//10)

        r_t = getattr(r_test_g, 'max')()
        r_c = getattr(r_control_g, 'max')()
        tit = 'mean, max'

    else:
        logger.debug("Plot all: median, perc90")
        r_t2 = getattr(r_test_g, 'median')()
        r_c2 = getattr(r_control_g, 'median')()
        x,y = getSortedCDF(r_t2)
        ax1.plot(x, y, marker='o', color='k', linestyle='--', label='250', markevery=len(y)//10)
        x,y = getSortedCDF(r_c2)
        ax1.plot(x, y, marker='d', color='r', linestyle='--', label='105', markevery=len(y)//10)

        r_t = getattr(r_test_g, 'quantile')(0.95)
        r_c = getattr(r_control_g, 'quantile')(0.95)

    x,y = getSortedCDF(r_t)
    ax1.plot(x, y, marker='o', color='b', label='250', markevery=len(y)//10)
    x,y = getSortedCDF(r_c)
    ax1.plot(x, y, marker='d', color='g', label='105', markevery=len(y)//10)

    ax1.set_xscale('log')
    ax1.set_xlabel("Prime-time Ratio")
    ax1.set_ylabel('CDF')
    ax1.set_title('Prime-time Ratio per Device - '+tit)

    filename_label=param.upper()
    plotname = 'prime-time-ratio-per-device-cdf-'+filename_label
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_cdf_all_bytes(test_full, control_full, PLOTPATH):
    fig1, ax1 = plt.subplots(1,1)

    x1, y1 = getSortedCDF( test_full['throughput'].values )
    ax1.plot(x1, y1, marker='o', color='b', markevery=len(y1)//10, linestyle='--', label='250')
    x2, y2 = getSortedCDF( control_full['throughput'].values )
    ax1.plot(x2, y2, marker='d', color='g', markevery=len(y2)//10, linestyle='--', label='105')

    #format_axes(ax1)
    ax1.set_xscale('log')
    ax1.set_xlabel("Data Rate [kbps]")
    ax1.set_ylabel('CDF')
    ax1.set_title('All Bytes')

    plotname = 'cdf-all-bytes'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE " + PLOTPATH + plotname)
    plt.close()
    return

def plot_cdf_max_per_device(test_full, control_full, PLOTPATH):
    ## CDF max per device
    ## 90perc per device
    fig1, ax1 = plt.subplots(1,1)
    ctr = 0
    c = ['b', 'g', 'k', 'r']
    m = ['o', 'd']
    lab = ['test', 'control']

    for df in [test_full, control_full]:
        #xdata = df.groupby('Device_number')['speed'].max()
        xdata = df.groupby('Device_number')['throughput'].max()
        xdata2 = df.groupby('Device_number')['throughput'].quantile(0.95)
        x,y = getSortedCDF(xdata)
        ax1.plot(x, y, color=c[ctr], marker=m[ctr], markevery=len(y)//10, label=lab[ctr]+'-max')
        x,y = getSortedCDF(xdata2)
        ax1.plot(x, y, color=c[ctr+2], marker=m[ctr], ls='--', markevery=len(y)//10, label=lab[ctr]+'-90%ile')
        ctr+=1
    #format_axes(ax1)
    ax1.set_xscale('log')
    ax1.set_xlabel("Data Rate [kbps]")
    ax1.set_ylabel('CDF')
    ax1.set_title('Max per Device')

    plotname = 'cdf-max-per-device'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_cdf_max_per_day_per_device(test_full, control_full, PLOTPATH):
    ## CDF max per day per device
    ## 90perc per day per device
    fig1, ax1 = plt.subplots(1,1)
    ctr = 0
    c = ['b', 'g', 'k', 'r']
    m = ['o', 'd']
    lab = ['test', 'control']

    for df in [test_full, control_full]:
        #xdata = df.groupby('Device_number')['speed'].max()
        xdata = df.groupby(['Device_number', 'date'])['throughput'].max()
        xdata2 = df.groupby(['Device_number', 'date'])['throughput'].quantile(0.95)
        x,y = getSortedCDF(xdata)
        ax1.plot(x, y, color=c[ctr], marker=m[ctr], markevery=len(y)//10, label=lab[ctr]+'-max')
        x,y = getSortedCDF(xdata2)
        ax1.plot(x, y, color=c[ctr+2], marker=m[ctr], ls='--', markevery=len(y)//10, label=lab[ctr]+'-90%ile')
        ctr+=1
    #format_axes(ax1)
    ax1.set_xscale('log')
    ax1.set_xlabel("Data Rate [kbps]")
    ax1.set_ylabel('CDF')
    ax1.set_title('Max per Day per Device')

    plotname = 'cdf-max-per-day-per-device'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

# for persistence and prevalance
def plot_prevalence_total_devices(test_full, control_full, PLOTPATH):

    maxThresh = max( test_full['throughput'].max(), control_full['throughput'].max() )
    minThresh = min( test_full['throughput'].min(), control_full['throughput'].min() )
    stepThresh = (maxThresh - minThresh)/20

    fig1, ax1 = plt.subplots(1,1)
    ctr = 0
    c = ['b', 'g']
    m = ['o', 'd']
    lab = ['test', 'control']
    for df in [test_full, control_full]:
        xdata = []
        ydata = []
        for THRESH in np.arange(minThresh, maxThresh, stepThresh):
            xdata.append(THRESH)
            sliced_df = df [ df['throughput'] >= THRESH ]
            num_dev = len( sliced_df['Device_number'].unique() )
            ydata.append( num_dev )
        ax1.plot(xdata, ydata, color=c[ctr], marker=m[ctr], label=lab[ctr])
        ctr+=1
    ax1.set_xscale('linear')
    ax1.set_xlabel('threshold [kbps]')
    ax1.set_ylabel('Number of Devices')
    ax1.set_yscale('log')
    ax1.set_title("Prevalence: total devices")

    plotname = 'slice-dataset-threshold-count-ndevices'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return
