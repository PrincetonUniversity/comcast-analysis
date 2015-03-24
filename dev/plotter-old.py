import pandas as pd
import csv
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime
import pickle as pkl
import os
import numpy as np
import logging
import matplotlib
# avoid xWindows errors in plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#PROCESSEDPATH='processed/'
DATAPATH='/data/users/sarthak/comcast-data/'
PROCESSEDPATH='/data/users/sarthak/comcast-data/processed/'
OUTPUTPATH='/data/users/sarthak/comcast-data/output/'

'''
Generate a test data/control set for quick play
1. test-data-set
head -300 250-test.dat > test-data-set.dat
tail -300 250-test.dat >> test-data-set.dat
2. test-control-set
head -300 control1.dat > test-control-set.dat
tail -300 control1.dat >> test-control-set.dat
'''

DATASET='250-test'
CONTROLSET='control1'

keys = ['Device_number',
         'end_time',
         'date_service_created',
         'service_class_name',
         'cmts_inet',
         'service_direction',
         'port_name',
         'octets_passed',
         'device_key',
         'service_identifier']
header_row = ['Device_number',
              'end_time',
              'cmts_inet',
              'service_direction',
              'octets_passed']

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')
DEBUG_ROW_DISPLAY = 100000      #display every 100,000th row in original data
DEBUG_LOG_COUNTER_DISPLAY = 100     #display every 100th device number


def read_dataframe(folder, filename):
    header_row = ['device_number', 'end_time', 'cmts_inet', 'direction', 'bytes']
    return pd.read_csv(PROCESSEDPATH + folder + '/' + filename, names=header_row)

# CDF plotter
def plotCDF_list(data, ax, c='blue'):
    '''
    data = list for which we need to plot cdf
    ax = axis
    '''
    sorted_data=np.sort( data )
    #logger.debug(sorted_data)
    #yvals=np.arange(len(sorted_data))/float(len(sorted_data))
    yvals=np.cumsum(sorted_data)
    #logger.debug(yvals)
    ax.plot( sorted_data, yvals, color=c )
    return

def plotCDF_counter(counter, ax, c='blue'):
    '''
    data = Counter object, not in increasing order
    ax = axis
    '''
    sorted_keys=sorted( counter )
    #log.debug(sorted_data)
    #yvals=np.arange(len(sorted_data))/float(len(sorted_data))
    yvals=np.cumsum([counter[k] for k in sorted_keys])
    #log.debug(yvals)
    ax.plot( sorted_keys, yvals, color=c, marker='o' )
    return


if __name__ == "__main__":

    # load pickles for plotting cdf
    # Load Data
    peakUsage = pkl.load(open(OUTPUTPATH+DATASET+"/"+'peakBytesPerUser.pkl', 'r'))
    noZeroHist = pkl.load(open(OUTPUTPATH+DATASET+"/"+'noZeroCounter.pkl', 'r'))
    peakHist = pkl.load(open(OUTPUTPATH+DATASET+"/"+'peakCounter.pkl', 'r'))
    peakUsage_control = pkl.load(open(OUTPUTPATH+CONTROLSET+"/"+'peakBytesPerUser.pkl', 'r'))
    noZeroHist_control = pkl.load(open(OUTPUTPATH+CONTROLSET+"/"+'noZeroCounter.pkl', 'r'))
    peakHist_control = pkl.load(open(OUTPUTPATH+CONTROLSET+"/"+'peakCounter.pkl', 'r'))

    logger.info("Plotting...")
    # plot double subplot CDFs for upstream and downstream
    # peak usage per user over all time
    fig0, ax0 = plt.subplots(2, 1, sharex=True)

    plotCDF_list(peakUsage['up'].values(), ax0[0])
    plotCDF_list(peakUsage_control['up'].values(), ax0[0], 'g')

    plotCDF_list(peakUsage['dw'].values(), ax0[1])
    plotCDF_list(peakUsage_control['dw'].values(), ax0[1], 'g')

    ax0[1].set_xlabel('Peak Bytes per Device Number [Bytes]')
    ax0[0].set_ylabel('CDF')
    ax0[1].set_ylabel('CDF')
    fig0.savefig(OUTPUTPATH + "peakBytesPerDeviceCDF")

    # No Zeros CDF
    fig1, ax1 = plt.subplots(2, 1)

    plotCDF_counter(noZeroHist['up'], ax1[0])
    plotCDF_counter(noZeroHist_control['up'], ax1[0], 'g')

    plotCDF_counter(noZeroHist['dw'], ax1[1])
    plotCDF_counter(noZeroHist_control['dw'], ax1[1], 'g')

    ax1[1].set_xlabel('Non Zero Bytes per Device time-slot [Bytes]')
    ax1[0].set_ylabel('CDF')
    ax1[1].set_ylabel('CDF')
    fig1.savefig(OUTPUTPATH + "noZeroCDF")
    #fig1.show()

    fig2, ax2 = plt.subplots(2, 1)

    plotCDF_counter(peakHist['up'], ax2[0])
    plotCDF_counter(peakHist_control['up'], ax2[0], 'g')

    plotCDF_counter(peakHist['dw'], ax2[1])
    plotCDF_counter(peakHist_control['dw'], ax2[1], 'g')

    ax2[1].set_xlabel('Peak Bytes per Device per Day [Bytes]')
    ax2[0].set_ylabel('CDF')
    ax2[1].set_ylabel('CDF')
    fig2.savefig(OUTPUTPATH + "peakBytesPerDayPerDeviceCDF")
    #fig2.show()
