from __future__ import division
import pickle as pkl
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

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
def plot_initial_timeseries(g1, g2):
    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].sum().plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, markeredgecolor='none', label='250Mbps')
    g2["throughput"].sum().plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('Throughput [Mbps]')
    ax1.set_title('Total Throughput')
    fig1.savefig('timeseries-throughput-TOTAL')

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].max().plot(ax=ax1, marker='o', alpha=.5, linestyle='', markersize=4, markeredgecolor='none',label='250Mbps')
    g2["throughput"].max().plot(ax=ax1, marker='d', alpha=.5, linestyle='', markersize=4, markeredgecolor='none',label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('Throughput [Mbps]')
    ax1.set_title('Max Throughput')
    fig1.savefig('timeseries-throughput-MAX')

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].median().plot(ax=ax1, marker='o', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='250Mbps')
    g2["throughput"].median().plot(ax=ax1, marker='d', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('Throughput [Mbps]')
    ax1.set_title('Median Throughput')
    fig1.savefig('timeseries-throughput-MEDIAN')

    fig1, ax1 = plt.subplots(1, 1, figsize=(18,5))
    g1["throughput"].mean().plot(ax=ax1, marker='o', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='250Mbps')
    g2["throughput"].mean().plot(ax=ax1, marker='d', alpha=.5, linestyle='',markersize=4, markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('Throughput [Mbps]')
    ax1.set_title('Mean Throughput')
    fig1.savefig('timeseries-throughput-MEAN')

    return

def plot_primetime_ratio_user(tps1, rperday1, rperdev1, tps2, rperday2, rperdev2, param='median'):

    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    rperday1.plot(ax=ax1, marker='o', linestyle='-', markeredgecolor='none', label='250Mbps')
    rperday2.plot(ax=ax1, marker='d', linestyle='-', markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('90%ile : '+param)
    ax1.set_yscale('log')
    ax1.set_title("mean [90%ile:"+param+"] ratio (per device)")
    fig1.savefig('ratio-timeseries-percentile-'+param)

    x1,y1 = getSortedCDF(rperdev1.values)
    x2,y2 = getSortedCDF(rperdev2.values)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    ax1.plot(x1, y1, marker='o', linestyle='-', markeredgecolor='none', label='250Mbps')
    ax1.plot(x2, y2, marker='d', linestyle='-', markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.grid(1)
    ax1.set_xlabel('mean ( 90%ile : '+param+' )')
    ax1.set_xscale('log')
    ax1.set_title("Distribution of device mean [90%ile:"+param+"] ratio (per day)")
    fig1.savefig('ratio-CDF-devices-percentile-'+param)

    x1,y1 = getSortedCDF( all_ratios(tps1, 'perc90', param).values )
    x2,y2 = getSortedCDF( all_ratios(tps2, 'perc90', param).values )
    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    ax1.plot(x1, y1, marker='o', linestyle='-', markeredgecolor='none', label='250Mbps')
    ax1.plot(x2, y2, marker='d', linestyle='-', markeredgecolor='none', label='100Mbps')
    ax1.legend(loc='best')
    ax1.grid(1)
    ax1.set_xlabel('90%ile : '+param)
    ax1.set_xscale('log')
    ax1.set_title("Distribution of [90%ile:"+param+"] ratio (ALL)")
    fig1.savefig('ratio-CDF-devices-percentile-'+param+'-ALL')
    return

def plot_primetime_ratio_isp(g1, g2):
    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))
    g1['throughput'].max().plot(ax=ax1, color='b', linestyle='-.', linewidth=3, label='250Mbps-max')
    g2['throughput'].max().plot(ax=ax1, color='g', linestyle='-.', linewidth=3, label='100Mbps-max')

    g1['throughput'].quantile(0.9).plot(ax=ax1, color='k', linestyle='-', label='250Mbps-perc90')
    g2['throughput'].quantile(0.9).plot(ax=ax1, color='r', linestyle='-', label='100Mbps-perc90')

    g1['throughput'].mean().plot(ax=ax1, color='b', linestyle=':', linewidth=3, label='250Mbps-mean')
    g2['throughput'].mean().plot(ax=ax1, color='g', linestyle=':', linewidth=3, label='100Mbps-mean')

    g1['throughput'].median().plot(ax=ax1, color='k', linestyle='--', linewidth=2, label='250Mbps-median')
    g2['throughput'].median().plot(ax=ax1, color='r', linestyle='--', linewidth=2, label='100Mbps-median')

    ax1.legend(loc='best')
    ax1.set_ylabel('Throughput [Mbps]')
#ax1.set_yscale('log')
    ax1.set_title("Throughput Statistics in 15 min slot")
    fig1.savefig('describe-total-throughput-per-day')
    return

def get_peak_primetime_per_set(test_full, control_full):
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

    df_primetime.to_pickle("df_best_primetime_hour.pkl")
    df_primetime.to_html("df_best_primetime_hour.html")
    return df_primetime

def plot_primetime_ratio_by_date(r_test, r_control):
    fig1, ax1 = plt.subplots(1, 1, figsize=(13,8))
    r_test.plot(ax=ax1, color='b', marker='o', linestyle='--',  label='250Mbps')
    r_control.plot(ax=ax1, color='g', marker='d', linestyle='-',  label='100Mbps')
    ax1.legend(loc='best')
    ax1.set_ylabel('Prime-time ratio')
    #ax1.set_yscale('log')
    ax1.set_title("avg throughput (peak hour) : avg throughput (non-peak hour)")
    fig1.savefig('prime-time-ratio-by-date')
    return

def plot_primetime_ratio_per_device(ratio, ratio2):
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
    fig1.savefig('cdf-prime-time-ratio-per-device')
    return

def plot_cdf_all_bytes(test_full, control_full):
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
    ax1.set_xlabel("Throughput [Mbps]")
    ax1.set_ylabel('CDF')
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('All Bytes')
    fig1.savefig('cdf-all-bytes')
    return

def plot_cdf_max_per_device(test_full, control_full):
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
    ax1.set_xlabel("Throughput [Mbps]")
    ax1.set_ylabel('CDF')
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('Max per Device')
    fig1.savefig('cdf-max-per-device')
    return

def plot_cdf_max_per_day_per_device(test_full, control_full):
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
    ax1.set_xlabel("Throughput [Mbps]")
    ax1.set_ylabel('CDF')
    ax1.grid(1)
    ax1.legend(loc='best')
    ax1.set_title('Max per Day per Device')
    fig1.savefig('cdf-max-per-day-per-device')
    return

# for persistence and prevalance
def plot_prevalence_total_devices():
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
    ax1.set_xlabel('threshold [Mbps]')
    ax1.set_ylabel('Number of Devices')
    ax1.set_yscale('log')
    ax1.set_title("Prevalence: total devices")
    fig1.savefig('slice-dataset-threshold-count-ndevices')
    return

def main():
    ORIGPATH = "../../dev/"

    for folder in os.listdir("../separated/"):
        CURPATH = "../separated/"+folder
        os.chdir(CURPATH)

        logger.debug("load test and control sets from " + CURPATH)
        test_full = pd.read_pickle("test.pkl")
        control_full = pd.read_pickle("control.pkl")
        test_full['time'] = test_full['datetime'].apply(lambda x: x.time())
        control_full['time'] = control_full['datetime'].apply(lambda x: x.time())
        test_full['date'] = test_full['datetime'].apply(lambda x: x.date())
        control_full['date'] = control_full['datetime'].apply(lambda x: x.date())

        logger.debug("plot initial time series")
        g1 = test_full.groupby("datetime")
        g2 = control_full.groupby("datetime")
        plot_initial_timeseries(g1, g2)

        for param in ["mean", "median"]:
            logger.debug("plot prime time ratio user "+param)
            tps1 = throughput_stats_per_device(test_full)
            rperday1 = mean_ratios_per_day(tps1, 'perc90',param)
            rperdev1 = mean_ratios_per_device(tps1, 'perc90',param)
            tps2 = throughput_stats_per_device(control_full)
            rperday2 = mean_ratios_per_day(tps2, 'perc90',param)
            rperdev2 = mean_ratios_per_device(tps2, 'perc90',param)
            plot_primetime_ratio_user(tps1, rperday1, rperdev1, tps2, rperday2, rperdev2, param)

        logger.debug("plot prime time ratio isp")
        t1 = test_full.groupby('datetime')['throughput'].sum().reset_index()
        t1['time'] = t1['datetime'].apply(lambda x: x.time())
        t2 = control_full.groupby('datetime')['throughput'].sum().reset_index()
        t2['time'] = t2['datetime'].apply(lambda x: x.time())
        g1 = t1.groupby('time')
        g2 = t2.groupby('time')
        plot_primetime_ratio_isp(g1, g2)

        logger.debug("get primetime at different times")
        df_primetime = get_peak_primetime_per_set(test_full, control_full)
        start_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['start_time']
        stop_time_t = df_primetime.sort('test_ratio', ascending=False).iloc[0]['stop_time']
        logger.info("test set " + str(start_time_t) +" "+ str(stop_time_t))
        start_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['start_time']
        stop_time_c = df_primetime.sort('control_ratio', ascending=False).iloc[0]['stop_time']
        logger.info("control set " + str(start_time_c) +" "+ str(stop_time_c))

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
        plot_primetime_ratio_by_date(r_test, r_control)

        logger.debug("plot primetime ratio per device")
        ratio = peak_t.groupby('Device_number')['throughput'].mean() / nonpeak_t.groupby('Device_number')['throughput'].mean()
        ratio2 = peak_c.groupby('Device_number')['throughput'].mean() / nonpeak_c.groupby('Device_number')['throughput'].mean()
        plot_primetime_ratio_per_device(ratio, ratio2)

        logger.debug("plot dataset throughput CDFs")
        plot_cdf_all_bytes(test_full, control_full)
        plot_cdf_max_per_device(test_full, control_full)
        plot_cdf_max_per_day_per_device(test_full, control_full)

        logger.debug("plot prevalance: total devices by threshold")
        maxThresh = max( test_full['throughput'].max(), control_full['throughput'].max() )
        minThresh = min( test_full['throughput'].min(), control_full['throughput'].min() )
        stepThresh = (maxThresh - minThresh)/20

        logger.debug("DONE (for now)")

        logger.debug("Back to folder "+ORIGPATH)
        os.chdir(ORIGPATH)

    return

if __name__ == "__main__":
    main()
