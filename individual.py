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

DATAPATH = 'data/'
PROCESSEDPATH = 'processed/'
OUTPUTPATH = 'output/'

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
CONTROLSET = 'control2.dat' #'test-control-set.dat'
DATALABEL = DATASET.split('.')[0]
CONTROLLABEL = CONTROLSET.split('.')[0]
LIST_OF_CONTROLLABELS = ['control'+str(l) for l in range(1,9)]
LIST_OF_CONTROLLABELS.remove('control3')         #causes errors unslicable
NDEVICES = 1000

dataframe = {}
sliced = {}
twodaysliced = {}
datatoplot = {}

# load 250-test and one control set
dataframe['test'] = pd.read_csv(DATAPATH + DATASET, delimiter='|', names=header_row)
dataframe['control'] = pd.read_csv(DATAPATH + CONTROLSET, delimiter='|', names=header_row)

# choose 1000 random test devices and save them
for name, df in dataframe.items():
    devices = df['Device_number'].unique()
    sliced_devices = random.sample(devices, NDEVICES)
    df_sliced = df[ df['Device_number'].isin(sliced_devices) ]
    df_sliced.to_pickle(PROCESSEDPATH+'slice/'+name+str(NDEVICES)+'.pkl')
    sliced[name] = df_sliced

# just analyze two days worth of complete data
date_start = '2014-10-11 00:00:00'
date_end = '2014-10-13 00:00:00'
for name, df in sliced.items():
    df2 = df [ (df['end_time'] >= date_start) & (df['end_time'] < date_end) ]
    dfname = name+str(NDEVICES)+date_start.split(" ")[0]+"_"+date_end.split(" ")[0]
    df2.to_pickle(PROCESSEDPATH+'slice/'+dfname)
    twodaysliced[name] = df2

# two day scatter plot with z=service_class_name
for name, df in twodaysliced.items():
    up = twodaysliced['test'][twodaysliced['test']['service_direction']== 2]
    [['Device_number', 'service_class_name', 'end_time', 'octets_passed']]
    dw = twodaysliced['test'][twodaysliced['test']['service_direction']== 1]
    [['Device_number', 'service_class_name', 'end_time', 'octets_passed']]
    datatoplot[name, 'up'] = up
    datatoplot[name, 'dw'] = dw

# plot
colorset = random.sample(matplotlib.colors.cnames.keys(), 20)
for key, df in datatoplot.items():
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
    ax1.set_ylabel('Bytes seen across 1000 devices')
    ax1.set_yscale('log')
    ax1.legend(loc='best', scatterpoints=1)
    ax1.grid(1)
    fig1.tight_layout()
    fig1.savefig(OUTPUTPATH + '/DETAIL/' + key[0] + '_' + key[1])

