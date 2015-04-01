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

CONVERT_OCTETS = 8 / (15 * 60 * 1024)       #kbps

INPUTPATH = "/data/users/sarthak/comcast-data/separated/"
OUTPUTPATH = "/data/users/sarthak/comcast-data/plots/"
PROCESSEDPATH = "/data/users/sarthak/comcast-data/process/"
if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)

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
    perc90_per_day = g['throughput'].apply(lambda x: np.percentile(x, 90))

    throughput_stats['perc90'] = perc90_per_day
    throughput_stats = throughput_stats.rename(columns={'datetime':'date'})

    return throughput_stats

def aggregate_octets_stats_per_datetime(df):
    """
    aggregate bytes across devices for each datetime
    return pandas dataframe; Index=datetime; Columns=[weekday, time, sum, len, perc90, median, max, min, std, ...]
    """

    # when applied on df.agg() this field gets renamed to '<lambda>'
    perc90 = lambda x: x.quantile(0.9)

    # dataframe indexed by datetime with aggregate columns
    g = df.groupby('datetime')
    agg_octets = g['octets_passed'].agg([np.sum, np.mean, np.median, perc90, max, min, np.std]).rename(columns={'<lambda>':'perc90'})

    # add corresponding timestamp and weekday to columns
    agg_octets['time'] = agg_octets.index.time
    agg_octets['weekday'] = agg_octets.index.weekday

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
        #ratios = g.quantile(0.9)
        ratios = getattr(g, 'quantile')(0.9)
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
        #ratios = g.quantile(0.9)
        ratios = getattr(g, 'quantile')(0.9)
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

def get_prime_time_ratios(octets_stats):
    """
    Input: pandas dataframe of octets_stats; Index = [datetime]; Columns = [weekday, time, sum]
    Output: pandas dataframe of ratio of throughput_stats columns. Reindexed. Columns = [date, peakratio]
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
        ts1 = getattr(g1['throughput'], 'quantile')(0.9)
        ts2 = getattr(g2['throughput'], 'quantile')(0.9)
    else:
        logger.warning("unknown parameter to plot throughput timeseries")
        plt.close()
        return

    # plot the time series
    ts1.plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, label='test')
    ts2.plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, label='control')

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
    rperday1.plot(ax=ax1, marker='o', linestyle='-', markeredgecolor='none', label='test')
    rperday2.plot(ax=ax1, marker='d', linestyle='-', markeredgecolor='none', label='control')

    filename_label = agg_param.upper()
    ax1.set_ylabel(agg_param+' peak-ratio')
    ax1.set_yscale('log')
    ax1.set_title("Daily peak-ratio aggregated "+agg_param+" over devices")

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
    ax1.plot(x1, y1, marker='o', linestyle='-', markeredgecolor='none', label='test')
    ax1.plot(x2, y2, marker='d', linestyle='-', markeredgecolor='none', label='control')

    filename_label = agg_param.upper()
    ax1.set_xlabel(agg_param + ' peak-ratio per device')
    ax1.set_xscale('log')
    ax1.set_title("Distribution of peak-ratio per device aggregated "+agg_param+" over days")

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
    Input: pandas groupedby; Index=['time', 'weekday']; Columns=[sum, perc90, max, min, median, ...]
    param_device: denotes the agg of bytes in a particular datetime over devices for the timeseries
    param_time: denotes the agg over [week, time] group: median, mean, perc90, max
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))

    if param_time in ['mean', 'max', 'min', 'median']:
        ts1 = getattr(g1[param_device], param_time)
        ts2 = getattr(g2[param_device], param_time)

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test')
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control')
    elif param_time == 'perc90':
        ts1 = g1[param_device].quantile(0.9)
        ts2 = g2[param_device],quantile(0.9)

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test')
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control')
    else:
        logger.debug("no param_time selected so plot 90-%ile, median, max, mean")

        g1[param_device].max().plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-max')
        g2[param_device].max().plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-max')

        g1[param_device].quantile(0.9).plot(ax=ax1, color='k', linestyle='-', label='test-perc90')
        g2[param_device].quantile(0.9).plot(ax=ax1, color='r', linestyle='-', label='control-perc90')

        g1[param_device].mean().plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='test-mean')
        g2[param_device].mean().plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='control-mean')

        g1[param_device].median().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='test-median')
        g2[param_device].median().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='control-median')

    ax1.set_ylabel(param_time + '$_{time}$ '+param_device+'$_{device}$ Bytes')
    ax1.set_title("Aggregate Bytes in a 15 min slot")

    plotname = 'describe-total-octets-per-day'
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
    Input: pandas groupedby; Index=['time', 'weekday']; Columns=[sum, perc90, max, min, median, ...]
    param_device: denotes the agg of bytes in a particular datetime over devices for the timeseries
    param_time: denotes the agg over [week, time] group: median, mean, perc90, max
    """

    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))

    if param_time in ['mean', 'max', 'min', 'median']:
        ts1 = getattr(g1[param_device], param_time) * CONVERT_OCTETS
        ts2 = getattr(g2[param_device], param_time) * CONVERT_OCTETS

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-'+param_time)
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-'+param_time)
    elif param_time == 'perc90':
        ts1 = g1[param_device].quantile(0.9)* CONVERT_OCTETS
        ts2 = g2[param_device],quantile(0.9)* CONVERT_OCTETS

        ts1.plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-'+param_time)
        ts2.plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-'+param_time)
    else:
        logger.debug("no param_time selected so plot 90-%ile, median, max, mean over time")

        (g1[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-max')
        (g2[param_device].max() * CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-max')

        (g1[param_device].quantile(0.9)* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='-', label='test-perc90')
        (g2[param_device].quantile(0.9)* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='-', label='control-perc90')

        (g1[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='test-mean')
        (g2[param_device].mean()* CONVERT_OCTETS).plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='control-mean')

        (g1[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='test-median')
        (g2[param_device].median()* CONVERT_OCTETS).plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='control-median')

    ax1.set_ylabel(param_time + '$_{time}$ '+param_device+'$_{device}$ Data Rate [kbps]')
    ax1.set_title("Aggregate Data Rate in a 15 min slot")

    plotname = 'describe-total-throughput-per-day'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def get_peak_primetime_per_set(test_full, control_full, PLOTPATH):
# get avg peak vs non peak throughput per day
    myData = defaultdict(list)

    for ctr in range(24):
        start_time = datetime.time(ctr,0)
        stop_time = datetime.time( (ctr+4) % 24 )

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

        avg_test_peak = peak_t.groupby('date').mean()
        avg_test_nonpeak = nonpeak_t.groupby('date').mean()

        avg_control_peak = peak_c.groupby('date').mean()
        avg_control_nonpeak = nonpeak_c.groupby('date').mean()
        myData['start_time'].append( start_time )
        myData['stop_time'].append( stop_time )
        myData['peak_t'].append( peak_t['throughput'].mean() )
        myData['nonpeak_t'].append( nonpeak_t['throughput'].mean() )
        myData['peak_c'].append( peak_c['throughput'].mean() )
        myData['nonpeak_c'].append( nonpeak_c['throughput'].mean() )

    df_primetime = pd.DataFrame(myData)
    df_primetime['test_ratio'] = df_primetime['peak_t']/df_primetime['nonpeak_t']
    df_primetime['control_ratio'] = df_primetime['peak_c']/df_primetime['nonpeak_c']

    df_primetime.to_pickle(PLOTPATH + "df_best_primetime_hour.pkl")
    df_primetime.to_html(PLOTPATH + "df_best_primetime_hour.html")
    return df_primetime

def plot_primetime_ratio_by_date(r_test, r_control, PLOTPATH):
    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))
    r_test.plot(ax=ax1, color='b', marker='o', linestyle='--',  label='test')
    r_control.plot(ax=ax1, color='g', marker='d', linestyle='-',  label='control')
    ax1.set_ylabel('Prime-time ratio')
    #ax1.set_yscale('log')
    ax1.set_title("avg throughput (peak hour) : avg throughput (non-peak hour)")

    plotname = 'prime-time-ratio-by-date'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_primetime_ratio_per_device(ratio, ratio2, PLOTPATH):
    fig1, ax1 = plt.subplots(1,1)
    x,y = getSortedCDF(ratio)
    ax1.plot(x, y, marker='o', label='test', markevery=len(y)/10)
    x,y = getSortedCDF(ratio2)
    ax1.plot(x, y, marker='d', label='control', markevery=len(y)/10)
    ax1.set_xscale('log')
    ax1.set_xlabel("Prime-time Ratio")
    ax1.set_ylabel('CDF')
    ax1.set_title('Prime-time Ratio per Device')

    plotname = 'cdf-prime-time-ratio-per-device'
    ax1.grid(1)
    ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(PLOTPATH + plotname)
    logger.info("CREATE FILE "+PLOTPATH + plotname)
    plt.close()
    return

def plot_cdf_all_bytes(test_full, control_full, PLOTPATH):
    fig1, ax1 = plt.subplots(1,1)
    ctr = 0
    c = ['b', 'g', 'k', 'r']
    m = ['o', 'd']
    lab = ['test', 'control']

    for df in [test_full, control_full]:
        #xdata = df.groupby('Device_number')['speed'].max()
        xdata = df['throughput']
        x,y = getSortedCDF(xdata)
        ax1.plot(x, y, color=c[ctr], marker=m[ctr], markevery=len(y)/10, label=lab[ctr])
        ctr+=1
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
    logger.info("CREATE FILE "+PLOTPATH + plotname)
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
        xdata2 = df.groupby('Device_number')['throughput'].quantile(0.9)
        x,y = getSortedCDF(xdata)
        ax1.plot(x, y, color=c[ctr], marker=m[ctr], markevery=len(y)/10, label=lab[ctr]+'-max')
        x,y = getSortedCDF(xdata2)
        ax1.plot(x, y, color=c[ctr+2], marker=m[ctr], ls='--', markevery=len(y)/10, label=lab[ctr]+'-90%ile')
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
        xdata2 = df.groupby(['Device_number', 'date'])['throughput'].quantile(0.9)
        x,y = getSortedCDF(xdata)
        ax1.plot(x, y, color=c[ctr], marker=m[ctr], markevery=len(y)/10, label=lab[ctr]+'-max')
        x,y = getSortedCDF(xdata2)
        ax1.plot(x, y, color=c[ctr+2], marker=m[ctr], ls='--', markevery=len(y)/10, label=lab[ctr]+'-90%ile')
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

def mp_plotter(folder):
    CURPATH = INPUTPATH + folder + '/'
    PLOTPATH = OUTPUTPATH + folder + '/'
    PROCPATH = PROCESSEDPATH + folder + '/'
    if not os.path.exists(PLOTPATH):
        os.makedirs(PLOTPATH)
    if not os.path.exists(PROCPATH):
        os.makedirs(PROCPATH)

    logger.debug("load test and control sets from " + CURPATH)
    test_full = pd.read_pickle(CURPATH + "test.pkl")
    control_full = pd.read_pickle(CURPATH + "control.pkl")

    # CHANGE TIMEZONE TO MST
    test_full['datetime']-=datetime.timedelta(hours=6)
    control_full['datetime']-=datetime.timedelta(hours=6)


    g1 = test_full.groupby("datetime")
    g2 = control_full.groupby("datetime")
    #g1 = test_full.groupby(["weekday", "time"])
    #g2 = control_full.groupby(["weekday", "time"])

    logger.debug("plot initial time series")
    for param in ['sum', 'max', 'perc90', 'mean', 'median']:
        plot_initial_timeseries(g1, g2, param, PLOTPATH)

    # throughput stats calculation per device per day
    if not os.path.isfile(PROCPATH + 'tps1.pkl'):
        logger.debug("Calculate throughput stats per device for test")
        tps1 = throughput_stats_per_device_per_date(test_full)
        tps1.to_pickle(PROCPATH + 'tps1.pkl')
        logger.debug("Calculate throughput stats per device for control")
        tps2 = throughput_stats_per_device_per_date(control_full)
        tps2.to_pickle(PROCPATH + 'tps2.pkl')
    else:
        logger.debug("Load throughput stats per device for test")
        tps1 = pd.read_pickle(PROCPATH + 'tps1.pkl')
        logger.debug("Load throughput stats per device for control")
        tps2 = pd.read_pickle(PROCPATH + 'tps2.pkl')

    # peak ratio (defined) =  [perc90 : median] of throughput (per day per device)
    # returns pandas dataframe [ Device_number | date | peakratio ]
    logger.debug("Calculate peak ratio = [perc90:median] throughput per date per device")
    peak_ratio1 = get_peak_ratios(tps1, 'perc90', 'median')
    peak_ratio2 = get_peak_ratios(tps2, 'perc90', 'median')
    del tps1, tps2

    # use peak_ratio['peakratio'] to get all ratios regardless of day/time
    logger.debug("plot peak ratio CDF of all")
    plot_peak_ratio_cdf(peak_ratio1['peakratio'], peak_ratio2['peakratio'], 'all', PLOTPATH)

    for agg_param in ["min", "mean", "median", "perc90", "max"]:
        peak_ratio_per_day1 = ratios_per_date(peak_ratio1, agg_param)
        peak_ratio_per_day2 = ratios_per_date(peak_ratio2, agg_param)
        logger.debug("plot peak ratio CDF aggregated over dates: filter by "+agg_param)
        plot_peak_ratio_timeseries(peak_ratio_per_day1, peak_ratio_per_day2, agg_param, PLOTPATH)

        peak_ratio_per_dev1 = ratios_per_device(peak_ratio1, agg_param)
        peak_ratio_per_dev2 = ratios_per_device(peak_ratio2, agg_param)
        logger.debug("plot peak ratio timeseries aggregated over devices: filter by "+agg_param)
        plot_peak_ratio_cdf(peak_ratio_per_dev1, peak_ratio_per_dev2, agg_param, PLOTPATH)

    del peak_ratio1, peak_ratio2
    del peak_ratio_per_day1, peak_ratio_per_day2
    del peak_ratio_per_dev1, peak_ratio_per_dev2

    # octets stats calculation per datetime aggregate
    if not os.path.isfile(PROCPATH + 'os1.pkl'):
        logger.debug("Calculate octets stats per datetime for test")
        os1 = aggregate_octets_stats_per_datetime(test_full)
        os1.to_pickle(PROCPATH + 'os1.pkl')
        logger.debug("Calculate octets stats per datetime for control")
        os2 = aggregate_octets_stats_per_datetime(control_full)
        os2.to_pickle(PROCPATH + 'os2.pkl')
    else:
        logger.debug("Load octets stats per datetime for test")
        os1 = pd.read_pickle(PROCPATH + 'os1.pkl')
        logger.debug("Load octets stats per datetime for control")
        os2 = pd.read_pickle(PROCPATH + 'os2.pkl')

    # group octets [max, min, median, perc90, len, std] by weekday and time
    # column to select from g1 and g2 groups, originally in os1 and os2

    # selecting sum here would create plots biased towards the set with more
    # devices, so should use mean to unbias that
    # can also try 'perc90' across all devices or 'median' across all devices
    # and then take mean or median when we fold on time
    g1 = os1.groupby(['time', 'weekday'])
    g2 = os2.groupby(['time', 'weekday'])

    # parameter to aggregate over devices
    param_device = 'mean'

    # parameter to aggregate over a week
    param_time = 'all'

    logger.debug("plot aggregated bytes per day")
    plot_octets_per_day(g1, g2, param_device, param_time, PLOTPATH)

    logger.debug("plot aggregated bytes per day")
    plot_throughput_per_day(g1, g2, param_device, param_time, PLOTPATH)

    del g1, g2, os1, os2

    """
    # prime time ratio = sum octets in peak hour : sum octets in off-peak hour
    # returns pandas dataframe [ Device_number | datetime (date only) | peakratio ]
    # prime time ratio calc per datetime
    #TODO get_prime_time_ratio()
    logger.debug("Calculate peak ratio = [perc90:median] throughput per date per device")
    peak_ratio1 = get_peak_ratios(tps1, 'perc90', 'median')
    peak_ratio2 = get_peak_ratios(tps2, 'perc90', 'median')
    del tps1, tps2
    """

    #TODO WEEKDAYS/HOLIDAYS/WEEKENDS SPLIT
    # GET date AND time:
    logger.debug("Shitty way of getting date and time for datasets")
    test_full['time'] = test_full['datetime'].apply(lambda x: x.time())
    control_full['time'] = control_full['datetime'].apply(lambda x: x.time())
    test_full['date'] = test_full['datetime'].apply(lambda x: x.date())
    control_full['date'] = control_full['datetime'].apply(lambda x: x.date())
    #test_full['weekday'] = test_full['datetime'].apply(lambda x: x.weekday())
    #control_full['weekday'] = control_full['datetime'].apply(lambda x: x.weekday())

    logger.debug("get primetime at different times")
    df_primetime = get_peak_primetime_per_set(test_full, control_full, PLOTPATH)
    start_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['start_time']
    stop_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("test set " + str(start_time_t) +" "+ str(stop_time_t))
    start_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['start_time']
    stop_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("control set " + str(start_time_c) +" "+ str(stop_time_c))
    del df_primetime

    df = test_full # Salt Lake City on PST => 7-11pm == 2-6 AM
    bool_series = (df['time'] < stop_time_t) & (df['time'] >= start_time_t)
    peak_t = df[ bool_series ]
    nonpeak_t = df[ ~ bool_series ]
    df = control_full # Unknown, so using maxes
    bool_series = (df['time'] < stop_time_c) & (df['time'] >= start_time_c)
    peak_c = df[ bool_series ]
    nonpeak_c = df[ ~ bool_series ]
    r_test = peak_t.groupby('date')['throughput'].mean() / nonpeak_t.groupby('date')['throughput'].mean()
    r_control = peak_c.groupby('date')['throughput'].mean() / nonpeak_c.groupby('date')['throughput'].mean()

    logger.debug("plot prime time ratio by date")
    plot_primetime_ratio_by_date(r_test, r_control, PLOTPATH)
    del r_test, r_control

    logger.debug("plot primetime ratio per device")
    ratio = peak_t.groupby('Device_number')['throughput'].mean() / nonpeak_t.groupby('Device_number')['throughput'].mean()
    ratio2 = peak_c.groupby('Device_number')['throughput'].mean() / nonpeak_c.groupby('Device_number')['throughput'].mean()
    plot_primetime_ratio_per_device(ratio, ratio2, PLOTPATH)
    del peak_t, peak_c, nonpeak_t, nonpeak_c

    logger.debug("plot dataset throughput CDFs")
    plot_cdf_all_bytes(test_full, control_full, PLOTPATH)
    plot_cdf_max_per_device(test_full, control_full, PLOTPATH)
    plot_cdf_max_per_day_per_device(test_full, control_full, PLOTPATH)

    logger.debug("plot prevalance: total devices by threshold")
    plot_prevalence_total_devices(test_full, control_full, PLOTPATH)

    logger.debug("DONE "+folder+" (for now)")
    return


def mp_plot_all():
    pool = mp.Pool(processes=12) #use 12 cores only
    for folder in os.listdir(INPUTPATH):
        pool.apply_async(mp_plotter, args=(folder,))
    pool.close()
    pool.join()
    return

def main(argv):
    #for folder in os.listdir("../separated/"):
    for folder in [argv]:
        mp_plotter(folder)
    return

def test():
    mp_plotter('test_dw')
    return


if __name__ == "__main__":
    print "INPUTPATH ", INPUTPATH
    print "OUTPUTPATH ", OUTPUTPATH
    print "PROCESSEDPATH ", PROCESSEDPATH
    print "folder = ", sys.argv[1]
    #test()
    main(sys.argv[1])
    #mp_plot_all()
