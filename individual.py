from __future__ import division
import pandas as pd
import numpy as np
import random, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter
from itertools import cycle
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

colorset = random.sample(matplotlib.colors.cnames.keys(), 20)
header_row = ['Device_number',
         'end_time',
         'date_service_created',
         'service_class_name',
         'cmts_inet',
         'service_direction',
         'port_name',
         'octets_passed',
         'device_key',
         'service_identifier']

DATASET = '250-test.dat' #'test-data-set.dat'
CONTROLSET = 'control5.dat' #'test-control-set.dat'
DATALABEL = DATASET.split('.')[0]
CONTROLLABEL = CONTROLSET.split('.')[0]
#LIST_OF_CONTROLLABELS = ['control'+str(l) for l in range(1,9)]
#LIST_OF_CONTROLLABELS.remove('control3')         #causes errors unslicable

NDEVICES = 100
YMAXLIM = 0
date_start = '2014-10-11 00:00:00'
date_end = '2014-10-13 00:00:00'

DATAPATH = 'data/'
PROCESSEDPATH = 'processed/'+CONTROLLABEL
OUTPUTPATH = 'output/'+CONTROLLABEL
if not os.path.exists(PROCESSEDPATH):
    os.makedirs(PROCESSEDPATH)
if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)

datatoplot = {}

def load_csv():
    dataframe = {}
    # load 250-test and one control set
    dataframe['test'] = pd.read_csv(DATAPATH + DATASET, delimiter='|', names=header_row)
    dataframe['control'] = pd.read_csv(DATAPATH + CONTROLSET, delimiter='|', names=header_row)
    return dataframe

def select_devices(dataframe, ndev=NDEVICES):
    sliced = {}
    # choose 1000 random test devices and save them
    for name, df in dataframe.items():
        devices = df['Device_number'].unique()
        sliced_devices = random.sample(devices, ndev)
        df_sliced = df[ df['Device_number'].isin(sliced_devices) ]
        sliced[name] = df_sliced
    return sliced

def slice_data(sliced, date_start, date_end):
    twodaysliced = {}
    # just analyze two days worth of complete data
    for name, df in sliced.items():
        df2 = df [ (df['end_time'] >= date_start) & (df['end_time'] < date_end) ]
        dfname = name+str(NDEVICES)+date_start.split(" ")[0]+"_"+date_end.split(" ")[0]
        twodaysliced[name] = df2
    return twodaysliced

def split_directions_plotting(twodaysliced):
    # two day scatter plot with z=service_class_name
    for name, df in twodaysliced.items():
        up = twodaysliced['test'][twodaysliced['test']['service_direction']== 2]
        [['Device_number', 'service_class_name', 'end_time', 'octets_passed']]
        dw = twodaysliced['test'][twodaysliced['test']['service_direction']== 1]
        [['Device_number', 'service_class_name', 'end_time', 'octets_passed']]
        datatoplot[name, 'up'] = up
        datatoplot[name, 'dw'] = dw
    return datatoplot

# plot
def plot_detailed_timeseries(dataplot):
    for key, df in datatoplot.items():
        logger.info("Plot " + str(key)+ "ALL detailed")

        colors = iter(colorset)
        fig1, ax1 = plt.subplots(1,1, figsize=(20,10))
        df['datetime'] = pd.to_datetime(df['end_time'])
        lines = []
        for service, df_temp in df.groupby('service_class_name'):
            x = list(df_temp['datetime'])
            y = list(df_temp['octets_passed'])
            # colorcode by 'service_class_name'
            C = colors.next()
            lines.append( ax1.scatter(x=x, y=y, c=C, alpha=0.4, label=service+': '+str(len(df_temp)) ) )
        ax1.set_xlim( [ df['datetime'].iloc[0], df['datetime'].iloc[-1] ] )
        ax1.set_xlabel('DateTime')
        #if YMAXLIM != 0:
        #    ax1.set_ylim( right=YMAXLIM )
        ax1.set_ylabel('Bytes seen across 1000 devices')
        ax1.set_yscale('log')
        ax1.legend(loc='best', scatterpoints=1)
        ax1.grid(1)
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH + '/DETAIL/' + key[0] + '_' + key[1])
    return

# aggregate analysis from 1000 sliced set
# plot SUM, MAX
def plot_agg_timeseries(dataplot, method):
    for key, df in datatoplot.items():
        logger.info("Plot " + str(key)+ " " + method)

        fig1, ax1 = plt.subplots(1,1, figsize=(20,10))

        colors = iter(colorset)
        df['datetime'] = pd.to_datetime(df['end_time'])
        lines = []
        g = df.groupby(['Device_number', 'datetime'], as_index=False)
        if method == 'sum':
            df_temp = g.sum()
            #YMAXLIM = df_temp['octets_passed'].max()
        elif method == 'max':
            df_temp = g.max()
        else:
            df_temp = g.quantile(method/100)      #can't quantile datetime
            #df_temp = g.apply(lambda x: np.percentile(x, 90))      #can't quantile datetime
            #g = df.groupby(['Device_number', 'datetime']).apply(lambda x:np.percentile(x, method))

        x = list(df_temp['datetime'])
        y = list(df_temp['octets_passed'])
        C = colors.next()

        ax1.scatter(x=x, y=y, c=C, alpha=0.4, label=method+': '+str(len(df_temp)) )

        ax1.set_xlim( [ df['datetime'].iloc[0], df['datetime'].iloc[-1] ] )
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Bytes seen across 1000 devices')
        ax1.set_yscale('log')
        ax1.legend(loc='best', scatterpoints=1)
        ax1.grid(1)
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH + '/DETAIL/' + key[0] + '_' + key[1] + '-'+method)
    return

def plot_all_service_timeseries(dataplot):
    for key, df in datatoplot.items():
        logger.info("Plot " + str(key)+ " each service: " + str( df['service_class_name'].unique() ))

        colors = iter(colorset)
        df['datetime'] = pd.to_datetime(df['end_time'])
        for service, df_temp in df.groupby('service_class_name'):
            fig1, ax1 = plt.subplots(1,1, figsize=(20,10))
            x = list(df_temp['datetime'])
            y = list(df_temp['octets_passed'])
            # colorcode by 'service_class_name'
            C = colors.next()

            ax1.scatter(x=x, y=y, c=C, alpha=0.4, label=service+': '+str(len(df_temp)) )

            ax1.set_xlim( [ df['datetime'].iloc[0], df['datetime'].iloc[-1] ] )
            ax1.set_xlabel('DateTime')
            ax1.set_ylabel('Bytes seen across 1000 devices')
            ax1.set_yscale('log')
            ax1.legend(loc='best', scatterpoints=1)
            ax1.grid(1)
            fig1.tight_layout()
            service_path = OUTPUTPATH + '/DETAIL/' + key[0] + '/'
            if not os.path.exists(service_path):
                os.makedir(service_path)
            fig1.savefig(service_path + key[0] + '_' + key[1] + '-' + service)
    return

if __name__ == '__main__':
    slicename = 'oneday'
    NDEVICES = 100
    date_start = '2014-11-04 00:00:00'
    date_end = '2014-11-05 00:00:00'
    if not os.path.isfile(PROCESSEDPATH+'slice/'+slicename):
        df = load_csv()
        df2 = select_devices(df, NDEVICES)
        df3 = slice_data(df2, date_start, date_end)

        pkl.dump(df3, open(PROCESSEDPATH+'slice/'+slicename, 'w'))
        logger.warning("Pickle dict of slice_data to "+PROCESSEDPATH+"slice/"+slicename)
    else:
        df3 = pkl.load(open(PROCESSEDPATH+'slice/'+slicename, 'r'))

    dataplot = split_directions_plotting(df3)
    plot_detailed_timeseries(dataplot)
    plot_all_service_timeseries(dataplot)
    plot_agg_timeseries(dataplot, 'sum')
    plot_agg_timeseries(dataplot, 'max')
    plot_agg_timeseries(dataplot, 90)
