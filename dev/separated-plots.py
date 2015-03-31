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

INPUTPATH = "/data/users/sarthak/comcast-data/separated/"
OUTPUTPATH = "/data/users/sarthak/comcast-data/plots/"
if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)
# CDF
def getSortedCDF(data):
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

# Prime Time Ratio
def throughput_stats_per_device(dfin):
    df = dfin.copy()
    g = df.set_index('datetime').groupby(["Device_number"])
    throughput_stats = g['throughput'].resample('D', how=[np.sum, len, max, min, np.median, np.mean, np.std])
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['time'] = df['datetime'].apply(lambda x: x.time())
    g = df.groupby(['Device_number', 'date'], as_index=False)
    perc90_per_day = g['throughput'].apply(lambda x: np.percentile(x, 90))
    throughput_stats['perc90'] = perc90_per_day
    return throughput_stats

def mean_ratios_per_day(throughput_stats, numer='perc90', denom='median'):
    """ mean over devices grouped by date """
    ratios = (throughput_stats[numer]/throughput_stats[denom]).reset_index().groupby('datetime')[0].mean()
    return ratios

def mean_ratios_per_device(throughput_stats, numer='perc90', denom='median'):
    """ mean over dates grouped by device """
    ratios = (throughput_stats[numer]/throughput_stats[denom]).reset_index().groupby('Device_number')[0].mean()
    return ratios

def all_ratios(throughput_stats, numer='perc90', denom='median'):
    """ all ratios regardlesss of device/date """
    ratios = (throughput_stats[numer]/throughput_stats[denom])
    return ratios

# Plotter - timeseries
def plot_initial_timeseries(g1, g2, PLOTPATH):
    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].sum().plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, markeredgecolor='none', label='test')
    g2["throughput"].sum().plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.set_ylabel('Data Rate [kbps]')
    ax1.set_title('Total Avg Data Rate')
    fig1.savefig(PLOTPATH +'timeseries-throughput-TOTAL')
    plt.close()

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].max().plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, markeredgecolor='none',label='test')
    g2["throughput"].max().plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, markeredgecolor='none',label='control')
    ax1.legend(loc='best')
    ax1.set_ylabel('Data Rate [kbps]')
    ax1.set_title('Max Avg Data Rate')
    fig1.savefig(PLOTPATH +'timeseries-throughput-MAX')
    plt.close()

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].median().plot(ax=ax1, marker='o', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='test')
    g2["throughput"].median().plot(ax=ax1, marker='d', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.set_ylabel('Data Rate [kbps]')
    ax1.set_title('Median Avg Data Rate')
    fig1.savefig(PLOTPATH +'timeseries-throughput-MEDIAN')
    plt.close()

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].mean().plot(ax=ax1, marker='o', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='test')
    g2["throughput"].mean().plot(ax=ax1, marker='d', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.set_ylabel('Data Rate [kbps]')
    ax1.set_title('Mean Avg Data Rate')
    fig1.savefig(PLOTPATH +'timeseries-throughput-MEAN')
    plt.close()

    return

def plot_primetime_ratio_user(tps1, rperday1, rperdev1, tps2, rperday2, rperdev2, param, PLOTPATH):

    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    rperday1.plot(ax=ax1, marker='o', linestyle='-', markeredgecolor='none', label='test')
    rperday2.plot(ax=ax1, marker='d', linestyle='-', markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.set_ylabel('90%ile : '+param)
    ax1.set_yscale('log')
    ax1.set_title("mean [90%ile:"+param+"] ratio (per device)")
    fig1.savefig(PLOTPATH +'ratio-timeseries-percentile-'+param)
    plt.close()

    x1,y1 = getSortedCDF(rperdev1.values)
    x2,y2 = getSortedCDF(rperdev2.values)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    ax1.plot(x1, y1, marker='o', linestyle='-', markeredgecolor='none', label='test')
    ax1.plot(x2, y2, marker='d', linestyle='-', markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.grid(1)
    ax1.set_xlabel('mean ( 90%ile : '+param+' )')
    ax1.set_xscale('log')
    ax1.set_title("Distribution of device mean [90%ile:"+param+"] ratio (per day)")
    fig1.savefig(PLOTPATH +'ratio-CDF-devices-percentile-'+param)
    plt.close()

    x1,y1 = getSortedCDF( all_ratios(tps1, 'perc90', param).values )
    x2,y2 = getSortedCDF( all_ratios(tps2, 'perc90', param).values )
    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    ax1.plot(x1, y1, marker='o', linestyle='-', markeredgecolor='none', label='test')
    ax1.plot(x2, y2, marker='d', linestyle='-', markeredgecolor='none', label='control')
    ax1.legend(loc='best')
    ax1.grid(1)
    ax1.set_xlabel('90%ile : '+param)
    ax1.set_xscale('log')
    ax1.set_title("Distribution of [90%ile:"+param+"] ratio (ALL)")
    fig1.savefig(PLOTPATH +'ratio-CDF-devices-percentile-'+param+'-ALL')
    plt.close()
    return

def plot_primetime_ratio_isp(g1, g2, PLOTPATH):
    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))
    g1['throughput'].max().plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='test-max')
    g2['throughput'].max().plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='control-max')

    g1['throughput'].quantile(0.9).plot(ax=ax1, color='k', linestyle='-', label='test-perc90')
    g2['throughput'].quantile(0.9).plot(ax=ax1, color='r', linestyle='-', label='control-perc90')

    g1['throughput'].mean().plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='test-mean')
    g2['throughput'].mean().plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='control-mean')

    g1['throughput'].median().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='test-median')
    g2['throughput'].median().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='control-median')

    ax1.legend(loc='best')
    ax1.set_ylabel('Data Rate [kbps]')
#ax1.set_yscale('log')
    ax1.set_title("Data Rate Statistics in 15 min slot")
    fig1.savefig(PLOTPATH +'describe-total-throughput-per-day')
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
    ax1.legend(loc='best')
    ax1.set_ylabel('Prime-time ratio')
    #ax1.set_yscale('log')
    ax1.set_title("avg throughput (peak hour) : avg throughput (non-peak hour)")
    fig1.savefig(PLOTPATH +'prime-time-ratio-by-date')
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
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('Prime-time Ratio per Device')
    fig1.savefig(PLOTPATH +'cdf-prime-time-ratio-per-device')
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
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('All Bytes')
    fig1.savefig(PLOTPATH +'cdf-all-bytes')
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
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('Max per Device')
    fig1.savefig(PLOTPATH +'cdf-max-per-device')
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
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('Max per Day per Device')
    fig1.savefig(PLOTPATH +'cdf-max-per-day-per-device')
    plt.close()
    return

# for persistence and prevalance
def plot_prevalence_total_devices(test_full, control_full, PLOTPATH):
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
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_xscale('linear')
    ax1.set_xlabel('threshold [kbps]')
    ax1.set_ylabel('Number of Devices')
    ax1.set_yscale('log')
    ax1.set_title("Prevalence: total devices")
    fig1.savefig(PLOTPATH +'slice-dataset-threshold-count-ndevices')
    plt.close()
    return

def main(argv):
    #for folder in os.listdir("../separated/"):
    for folder in [argv]:
        mp_plotter(folder)
    return

def mp_plotter(folder):
    CURPATH = INPUTPATH +folder+"/"
    PLOTPATH = OUTPUTPATH + folder + '/'
    if not os.path.exists(PLOTPATH):
        os.makedirs(PLOTPATH)

    logger.debug("load test and control sets from " + CURPATH)
    test_full = pd.read_pickle(CURPATH + "test.pkl")
    control_full = pd.read_pickle(CURPATH + "control.pkl")

    # CHANGE TIMEZONE TO MST
    test_full['datetime']-=datetime.timedelta(hours=6)
    control_full['datetime']-=datetime.timedelta(hours=6)

    #TODO WEEKDAYS/HOLIDAYS/WEEKENDS SPLIT
    # GET date AND time:
    test_full['time'] = test_full['datetime'].apply(lambda x: x.time())
    control_full['time'] = control_full['datetime'].apply(lambda x: x.time())
    test_full['date'] = test_full['datetime'].apply(lambda x: x.date())
    control_full['date'] = control_full['datetime'].apply(lambda x: x.date())
    test_full['weekday'] = test_full['datetime'].apply(lambda x: x.weekday())
    control_full['weekday'] = control_full['datetime'].apply(lambda x: x.weekday())

    logger.debug("plot initial time series by week")
    #g1 = test_full.groupby("datetime")
    #g2 = control_full.groupby("datetime")
    g1 = test_full.groupby(["weekday", "time"])
    g2 = control_full.groupby(["weekday", "time"])
    plot_initial_timeseries(g1, g2, PLOTPATH)

    for param in ["mean", "median"]:
        logger.debug("plot prime time ratio user "+param)
        tps1 = throughput_stats_per_device(test_full)
        rperday1 = mean_ratios_per_day(tps1, 'perc90',param)
        rperdev1 = mean_ratios_per_device(tps1, 'perc90',param)
        tps2 = throughput_stats_per_device(control_full)
        rperday2 = mean_ratios_per_day(tps2, 'perc90',param)
        rperdev2 = mean_ratios_per_device(tps2, 'perc90',param)
        plot_primetime_ratio_user(tps1, rperday1, rperdev1, tps2, rperday2, rperdev2, param, PLOTPATH)
    del tps1, tps2, rperdev1, rperdev2, rperday1, rperday2

    logger.debug("plot prime time ratio isp")
    t1 = test_full.groupby('datetime')['throughput'].sum().reset_index()
    t1['time'] = t1['datetime'].apply(lambda x: x.time())
    t2 = control_full.groupby('datetime')['throughput'].sum().reset_index()
    t2['time'] = t2['datetime'].apply(lambda x: x.time())
    g1 = t1.groupby('time')
    g2 = t2.groupby('time')
    plot_primetime_ratio_isp(g1, g2, PLOTPATH)
    del g1, g2, t1, t2

    logger.debug("get primetime at different times")
    df_primetime = get_peak_primetime_per_set(test_full, control_full, PLOTPATH)
    start_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['start_time']
    stop_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("test set " + str(start_time_t) +" "+ str(stop_time_t))
    start_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['start_time']
    stop_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['stop_time']
    logger.info("control set " + str(start_time_c) +" "+ str(stop_time_c))
    del df_primetime

    logger.debug("plot prime time ratio by date")
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

    #logger.debug("plot prevalance: total devices by threshold")
    #maxThresh = max( test_full['throughput'].max(), control_full['throughput'].max() )
    #minThresh = min( test_full['throughput'].min(), control_full['throughput'].min() )
    #stepThresh = (maxThresh - minThresh)/20
    #TODO

    logger.debug("DONE "+folder+" (for now)")
    return


def mp_plot_all():
    pool = mp.Pool(processes=13) #use 13 cores only
    for folder in os.listdir(INPUTPATH):
        pool.apply_async(mp_plotter, args=(folder,))
    pool.close()
    pool.join()
    return


if __name__ == "__main__":
    #print sys.argv[1]
    #main(sys.argv[1])
    mp_plot_all()
