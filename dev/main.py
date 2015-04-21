from __future__ import division
import pandas as pd
import pickle as pkl
import numpy as np
import os, sys
import multiprocessing as mp
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

from plotter import *

CHANGE_TIMEZONE = 0     # for final_dw and final_up convert time during sanitization
INPUTPATH = "/data/users/sarthak/comcast-data/separated/"
OUTPUTPATH = "/data/users/sarthak/comcast-data/plots/"
#OUTPUTPATH = "/data/users/sarthak/comcast-analysis/plots/"
#OUTPUTPATH = "~/public_html/files/comcast/plots/"
PROCESSEDPATH = "/data/users/sarthak/comcast-data/process/"
if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)

# TODO make this class: easier to manage const
def init_setup(folder):
    CURPATH = INPUTPATH + folder + '/'
    PLOTPATH = OUTPUTPATH + folder + '/'
    PROCPATH = PROCESSEDPATH + folder + '/'
    if not os.path.exists(PLOTPATH):
        os.makedirs(PLOTPATH)
    if not os.path.exists(PROCPATH):
        os.makedirs(PROCPATH)

    logger.debug("load test and control sets from " + CURPATH)
    try:
        test_full = pd.read_pickle(CURPATH + "test.pkl")
        control_full = pd.read_pickle(CURPATH + "control.pkl")
    except Exception:
        logger.error(INPUTPATH + folder + " doesn't have the files needed")
        raise

    # CHANGE TIMEZONE TO MST
    if CHANGE_TIMEZONE:
        test_full['datetime']-=datetime.timedelta(hours=6)
        control_full['datetime']-=datetime.timedelta(hours=6)

    # Add date and time
    logger.info("Add the time column for datasets")
    if 'time' not in test_full.columns:
        test_full['time'] = test_full.set_index('datetime').index.time
        #test_full['time'] = test_full.set_index('datetime').resample('H').index.time
    if 'time' not in control_full.columns:
        control_full['time'] = control_full.set_index('datetime').index.time
    if 'date' not in test_full.columns:
        #test_full['date'] = test_full.set_index('datetime').resample('D').index
        test_full['date'] = test_full.set_index('datetime').index.date
    if 'date' not in control_full.columns:
        control_full['date'] = control_full.set_index('datetime').index.date
    logger.info("Done adding time column for datasets")

    return CURPATH, PLOTPATH, PROCPATH, test_full, control_full

def primetime(test_full, control_full, PLOTPATH):
    #mp_plotter('test_dw')

    logger.debug("get average (summed) peak primetime at different times")
    peak_t, nonpeak_t, peak_c, nonpeak_c = get_peak_nonpeak_series(test_full, control_full, PLOTPATH)

    field = 'octets_passed'
    logger.debug("get primetime ratio using field="+field)
    r_test, r_control = get_primetime_ratio(peak_t, nonpeak_t, peak_c, nonpeak_c)
    del peak_t, nonpeak_t, peak_c, nonpeak_c

    #logger.debug("draw a scatter plot of device vs datetime with colormap for ratio")
    #plot_primetime_ratio_scatter(r_test, r_control, PLOTPATH)

    param='all1'
    logger.debug("plot prime time ratio by date, group devices by "+param)
    plot_primetime_ratio_by_date(r_test, r_control, param, PLOTPATH)
    logger.debug("plot primetime ratio per device, group dates by "+param)
    plot_primetime_ratio_per_device(r_test, r_control, param, PLOTPATH)

    param = 'all2'
    logger.debug("plot prime time ratio by date, group devices by "+param)
    plot_primetime_ratio_by_date(r_test, r_control, param, PLOTPATH)
    logger.debug("plot primetime ratio per device, group dates by "+param)
    plot_primetime_ratio_per_device(r_test, r_control, param, PLOTPATH)
    del r_test, r_control

    return

def initial_timeseries(test_full, control_full, PLOTPATH):
    g1 = test_full.groupby("datetime")
    g2 = control_full.groupby("datetime")

    logger.debug("plot initial time series")
    for param in ['sum', 'max', 'perc90', 'mean', 'median']:
        plot_initial_timeseries(g1, g2, param, PLOTPATH)
    del g1, g2
    return

def peak_ratio(test_full, control_full, PROCPATH, PLOTPATH):
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
    logger.debug("Calculate peak ratio = [perc95:mean] throughput per date per device")
    peak_ratio1 = get_peak_ratios(tps1, 'perc90', 'mean')
    peak_ratio2 = get_peak_ratios(tps2, 'perc90', 'mean')

    #logger.debug("Calculate peak ratio = [perc95:median] throughput per date per device")
    #peak_ratio1 = get_peak_ratios(tps1, 'perc90', 'median')
    #peak_ratio2 = get_peak_ratios(tps2, 'perc90', 'median')
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

    return

def throughput_weekday(test_full, control_full, PROCPATH, PLOTPATH):

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
    g1 = os1.groupby([ 'day', 'time'])
    g2 = os2.groupby([ 'day', 'time'])

    # parameter to aggregate over devices
    param_device = 'mean'

    # parameter to aggregate over a week
    param_time = 'all'

    logger.debug("plot aggregated bytes throughput medians, perc95 per day")
    param_time = 'all1'
    plot_octets_per_day(g1, g2, param_device, param_time, PLOTPATH)
    plot_throughput_per_day(g1, g2, param_device, param_time, PLOTPATH)

    logger.debug("plot aggregated bytes + throughput max, mean per day")
    param_time = 'all2'
    plot_octets_per_day(g1, g2, param_device, param_time, PLOTPATH)
    plot_throughput_per_day(g1, g2, param_device, param_time, PLOTPATH)


    del g1, g2, os1, os2

    return

def plot_cdf(test_full, control_full, PLOTPATH):
    logger.debug("plot dataset throughput CDFs")
    plot_cdf_all_bytes(test_full, control_full, PLOTPATH)
    plot_cdf_max_per_device(test_full, control_full, PLOTPATH)
    plot_cdf_max_per_day_per_device(test_full, control_full, PLOTPATH)
    return

def prevalence(test_full, control_full, PLOTPATH):
    logger.debug("plot prevalance: total devices by threshold")
    plot_prevalence_total_devices(test_full, control_full, PLOTPATH)
    return

def mp_plotter(folder):
    """
    Parallelized version of plotter
    """

    CURPATH, PLOTPATH, PROCPATH, test_full, control_full = init_setup(folder)

    jobs = []

    jobs.append( mp.Process(target= initial_timeseries,
                            args=(test_full, control_full, PLOTPATH,)) )
    jobs.append( mp.Process(target= peak_ratio,
                            args=(test_full, control_full, PROCPATH, PLOTPATH,)) )
    jobs.append( mp.Process(target= primetime,
                            args=(test_full, control_full, PLOTPATH,)) )
    jobs.append( mp.Process(target= throughput_weekday,
                            args=(test_full, control_full, PROCPATH, PLOTPATH,)) )
    jobs.append( mp.Process(target= plot_cdf,
                            args=(test_full, control_full, PLOTPATH,)) )
    jobs.append( mp.Process(target= prevalence,
                            args=(test_full, control_full, PLOTPATH,)) )

    logger.debug("Start parallel code for folder "+folder)
    for proc in jobs:
        proc.start()

    return

def plotter(folder):

    # INITIALIZE
    CURPATH, PLOTPATH, PROCPATH, test_full, control_full = init_setup(folder)

    # TIME SERIES
    initial_timeseries(test_full, control_full, PLOTPATH)

    # PEAK RATIO
    peak_ratio(test_full, control_full, PLOTPATH)

    # PRIME TIME
    primetime(test_full, control_full, PLOTPATH)

    # THROUGHPUT PER WEEKDAY
    throughput_weekday(test_full, control_full, PLOTPATH)

    # PLOT CDF
    plot_cdf(test_full, control_full, PLOTPATH)

    # PREVALENCE
    prevalence(test_full, control_full, PLOTPATH)

    logger.debug("DONE "+folder+" (for now)")

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
    #logger.debug("Shitty way of getting date and time for datasets")
    #test_full['time'] = test_full.set_index('datetime').index.time
    #control_full['time'] = control_full.set_index('datetime').index.time
    #test_full['time'] = test_full['datetime'].apply(lambda x: x.time())
    #control_full['time'] = control_full['datetime'].apply(lambda x: x.time())
    #test_full['date'] = test_full['datetime'].apply(lambda x: x.date())
    #control_full['date'] = control_full['datetime'].apply(lambda x: x.date())
    #test_full['weekday'] = test_full['datetime'].apply(lambda x: x.weekday())
    #control_full['weekday'] = control_full['datetime'].apply(lambda x: x.weekday())


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
    folder = 'control1_dw'
    CURPATH, PLOTPATH, PROCPATH, test_full, control_full = init_setup(folder)
    primetime(test_full, control_full, PLOTPATH)
    return

if __name__ == "__main__":
    print "INPUTPATH ", INPUTPATH
    print "OUTPUTPATH ", OUTPUTPATH
    print "PROCESSEDPATH ", PROCESSEDPATH
    print "folder = ", sys.argv[1]
    #test()
    main(sys.argv[1])
    #mp_plot_all()
