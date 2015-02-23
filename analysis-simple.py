from __future__ import division
import pandas as pd
import numpy as np
import random, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter
import logging

# original 250-test.dat size is 95,000,000 rows
# head -1000000 250-test.dat > test-data-set.dat
# tail -1000000 250-test.dat >> test-data-set.dat
# head -1000000 control1.dat > test-control-set.dat
# tail -1000000 control1.dat >> test-control-set.dat
# when running in /data/users/sarthak it takes less than a min to load full

DATASET = '250-test.dat' #'test-data-set.dat'
CONTROLSET = 'control1.dat' #'test-control-set.dat'
DATALABEL = DATASET.split('.')[0]
CONTROLLABEL = CONTROLSET.split('.')[0]
NDEVICES = 1000
LIST_OF_CONTROLLABELS = ['control'+str(l) for l in range(1,9)]

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

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

#############################################################################
# load dataframes
#############################################################################

def load_csv():
    df1 = pd.read_csv(DATASET, delimiter='|', names=header_row)
    df2 = pd.read_csv(CONTROLSET, delimiter='|', names=header_row)
    return df1, df2

def filter_csv(df, dfname):
    df_up = df[df['service_direction'] == 2][['Device_number', 'end_time',
                                              'octets_passed']]
    df_dw = df[df['service_direction'] == 1][['Device_number', 'end_time',
                                              'octets_passed']]
    df_up.to_pickle(PROCESSEDPATH+dfname+'_up.pkl')
    df_dw.to_pickle(PROCESSEDPATH+dfname+'_dw.pkl')
    return

def load_dataframe(dfname):
    df_up = pd.read_pickle(PROCESSEDPATH+dfname+'_up.pkl')
    df_dw = pd.read_pickle(PROCESSEDPATH+dfname+'_dw.pkl')
    return df_up, df_dw

def save_dfs():
    df1, df2 = load_csv()
    filter_csv(df1, DATALABEL)
    filter_csv(df2, CONTROLLABEL)
    return

#############################################################################
# Helper func
#############################################################################
def getSortedCDF(data):
    """
    return x,y for a cdf given one dimensional unsorted data x
    """
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

#############################################################################
# Analysis
#############################################################################

class DataProcessor:
    def __init__(self, DFNAME):
        self.dfname = DFNAME
        return

    def set_dfname(self, DFNAME):
        self.dfname = DFNAME
        return

    def load_csv(self):
        return pd.read_csv(self.dfname + '.dat', delimiter='|',
                           names=header_row)

    def splitDateTime(self, df_temp):
        """
        dataframes are passed by reference by default.
        """
        df_temp['date'] = df_temp['end_time'].apply(lambda x:
                                                        x.split(" ")[0])
        df_temp['time'] = df_temp['end_time'].apply(lambda x:
                                                        x.split(" ")[1])
        return

    def save_pkl(self):
        df = self.load_csv()
        df_up = df[df['service_direction'] == 2][['Device_number', 'end_time',
                                                  'octets_passed']]
        df_dw = df[df['service_direction'] == 1][['Device_number', 'end_time',
                                                  'octets_passed']]
        self.splitDateTime(df_up)
        self.splitDateTime(df_dw)
        df_up.to_pickle(PROCESSEDPATH + self.dfname + '_up.pkl')
        df_dw.to_pickle(PROCESSEDPATH + self.dfname + '_dw.pkl')
        return df_up, df_dw

##############################################################################

def meta_data():
    listDS = ['test-data-set', 'test-control-set', '250-test', 'control1',
              'control2', 'control3' ,'control4', 'control5', 'control6',
              'control7', 'control8']
    attrib = defaultdict(list)
    for dfname in listDS:
        for direction in ['up', 'dw']:
            logger.info('load dataframe ' + dfname + '_' + direction)

            df = pd.read_pickle(PROCESSEDPATH + dfname +'_'+ direction +'.pkl')

            attrib['name'].append(dfname)
            attrib['direction'].append(direction)
            attrib['len'].append(len(df))
            df.sort('end_time', ascending=True)
            attrib['date_start'].append(df['end_time'].iloc[0])
            attrib['date_end'].append(df['end_time'].iloc[-1])
            devices = np.sort(df['Device_number'].unique())
            attrib['num_devices'].append(len(devices))
            attrib['device_start'].append(devices[0])
            attrib['device_end'].append(devices[-1])
            attrib['max_bytes'].append(df['octets_passed'].max())
            attrib['mean_bytes'].append(df['octets_passed'].mean())
            attrib['median_bytes'].append(df['octets_passed'].median())
            attrib['perc90_bytes'].append( np.percentile( df['octets_passed'],
                                                         90) )
            logger.debug(pd.DataFrame(attrib))
    df_attributes = pd.DataFrame(attrib)[['name', 'direction', 'len',
                                          'date_start', 'date_end',
                                          'num_devices', 'device_start',
                                          'device_end', 'max_bytes',
                                          'mean_bytes', 'median_bytes',
                                          'perc90_bytes']]
    df_attributes.to_pickle(OUTPUTPATH + 'data_attributes.pkl')
    return df_attributes

##############################################################################
# SLICE data
# 1000 devices, 9/30 -- 12/25
##############################################################################

def get_dates(df):
    """
    given df, sort dates and get date_start and date_end
    """
    df.sort('end_time', ascending=True)
    date_start = df['end_time'].iloc[0]
    date_end = df['end_time'].iloc[-1]
    return date_start, date_end

def slice_df(df, num_devices, date_start, date_end):
    """
    num_devices = 1000
    date_start = 2014-09-30 00:00:00
    date_end = 2014-12-25 20:45:00
    """
    df2 = df[ (df['end_time'] >= date_start) & (df['end_time'] <= date_end) ]
    devices = df2['Device_number'].unique()
    sliced_devices = random.sample(devices, num_devices)
    df3 = df2[ df2['Device_number'].isin(sliced_devices) ]
    return df3

def slicer():
    """
    THIS IS NOT THE RIGHT WAY...
    compose and pickle 4 sets of 1000 devices and same date range
    TEST_up, TEST_dw, CONTROL_up, CONTROL_dw
    see output/data_attributes.html for details
    250-test -> 1000, daterange
    control1 -> 1000, daterange
    control2 -> 1000, daterange
    control3 -> XXX
    control4 -> 1000, ['2014-10-29 16:30:00' : enddate]
    control5 -> 1000, daterange
    control6 -> 1000, daterange
    control7 -> 1000, daterange
    control8 -> 1000, daterange
    """

    NDEVICES = 1000
    DSTART = '2014-09-30 00:00:00'
    DEND = '2014-12-25 20:45:00'

    # TEST SET
    dfname = '250-test'
    direction = 'up'
    df = pd.read_pickle(PROCESSEDPATH + dfname +'_'+ direction +'.pkl')
    df_temp = slice_df(df, NDEVICES, DSTART, DEND)
    df_temp['dataset'] = dfname
    df_temp.to_pickle(PROCESSEDPATH + 'TEST_up.pkl')

    direction = 'dw'
    df = pd.read_pickle(PROCESSEDPATH + dfname +'_'+ direction +'.pkl')
    df_temp = slice_df(df, NDEVICES, DSTART, DEND)
    df_temp['dataset'] = dfname
    df_temp.to_pickle(PROCESSEDPATH + 'TEST_dw.pkl')

    # CONTROL SET
    CONTROL = []
    direction = 'up'
    for dfname in ['control1', 'control2', 'control5', 'control6', 'control7',
                   'control8']:
        df = pd.read_pickle(PROCESSEDPATH + dfname +'_'+ direction +'.pkl')
        if dfname == 'control4':
            df_temp = slice_df(df, NDEVICES, '2014-10-29 16:30:00', DEND)
        else:
            df_temp = slice_df(df, NDEVICES, DSTART, DEND)
        df_temp['dataset'] = dfname
        CONTROL.append(df_temp)
    dfname = 'control4'
    pd.concat(CONTROL).to_pickle(PROCESSEDPATH + 'TEST_up.pkl')

    #TODO incomplete

    return


##############################################################################

class Plotter:
    def __init__(self, dlabel=DATALABEL, clabel=CONTROLLABEL, XSCALE='log'):
        self.test_up = 0
        self.test_dw = 0
        self.control_up = 0
        self.control_dw = 0
        self.DLABEL = dlabel
        self.CLABEL = clabel
        self.DISPLAY_MEAN = 1
        self.OUTPUTPATH = OUTPUTPATH
        self.XSCALE = XSCALE
        return

    def load_df(self):
        self.test_up, self.test_dw = load_dataframe(self.DLABEL)
        self.control_up, self.control_dw = load_dataframe(self.CLABEL)
        return

    def import_df(self, test_up, test_dw, control_up, control_dw):
        self.test_up = test_up
        self.test_dw = test_dw
        self.control_up = control_up
        self.control_dw = control_dw
        return

    def allBytes(self, df, method):
        """
        return df, xlabel, filename
        """
        dfr = df['octets_passed']
        return dfr, "All Seen Bytes","CDF-AllBytes"

    def thresholdBytes(self, df, threshold=0):
        """
        return df, xlabel, filename
        """
        dfr = df[df['octets_passed'] > threshold]['octets_passed']
        return dfr, "Bytes > "+str(threshold),"CDF-ThresholdBytes-"+str(threshold)

    def getBytesPerDevice(self, df, method='Peak'):
        """
        Peak: max()
        Median: median()
        90 percentile: quantile(0.90)
        return df, xlabel, filename
        """
        if method=='Peak':
            dfr = df.sort('octets_passed', ascending=False).groupby(
                ['Device_number'], as_index=False).first()['octets_passed']
        elif method=='Median':
            dfr = df.groupby(['Device_number'],
                             as_index=False).median()['octets_passed']
        elif method=='90perc':
            dfr = df.groupby(['Device_number'],
                             as_index=False).quantile(.90)['octets_passed']
        else:
            dfr = None
        return dfr, method+" Bytes per Device","CDF-BytesPerDevice-"+method

    def getBytesPerDevicePerDay(self, df, method='Peak'):
        """
        return df, xlabel, filename
        """
        if method=='Peak':
            #dfr = df.sort('octets_passed', ascending=False).groupby(
            #    ['Device_number', 'date'], as_index=False).first()['octets_passed']
            dfr = df.groupby(['Device_number', 'date'],
                             as_index=False).max()['octets_passed']
        elif method=='Median':
            dfr = df.groupby(['Device_number', 'date'],
                             as_index=False).median()['octets_passed']
        elif method=='90perc':
            dfr = df.groupby(['Device_number', 'date'],
                             as_index=False).quantile(.90)['octets_passed']
        else:
            dfr = None
        return dfr, method+" Bytes per Device per Day",\
                "CDF-BytesPerDevicePerDay-"+method

    def getBytesPerDevicePerTimeSlot(self, df, method='Peak'):
        """
        return df, xlabel, filename
        """
        if method=='Peak':
            dfr = df.sort('octets_passed', ascending=False).groupby(
                ['Device_number', 'time'], as_index=False).first()['octets_passed']
        elif method=='Median':
            dfr = df.groupby(['Device_number', 'time'],
                             as_index=False).median()['octets_passed']
        #elif method=='90perc':
        #    logger.debug("plot "+method+" bytes per device per time slot.\nDF:")
        #    dfr = df.groupby(['Device_number', 'time'],
        #                     as_index=False).quantile(.90)['octets_passed']
        #    logger.debug(dfr.describe())
        else:
            dfr = None
        return dfr, method+" Bytes per Device per Time-Slot",\
                "CDF-BytesPerDevicePerTimeSlot-"+method

    def CDFPlotter(self, callback, method='Peak'):
        """
        Plot 2 subplot CDF uplink and downlink with test (b) and control (g)
        data. The data itself is calculated by callback. Options: allBytes,
        nonZeroBytes, peakBytesPerDevice, peakBytesPerDevicePerDay,
        peakBytesPerDevicePerTimeSlot
        Each df (testUP, testDW, controlUP, controlDW) is filtered according
        to the callback criteria, and getCDF() is used to sort xvals and
        and calculate appropriate yvals in range.
        TODO: add marker intervals to plotter
        """
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))

        labels = iter([self.DLABEL+'_up',self.CLABEL+'_up',self.DLABEL+'_dw',
                       self.CLABEL+'_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])
        markers = iter(['o','x','o','x'])

        for df_temp in [self.test_up, self.control_up, self.test_dw,
                        self.control_dw]:
            df, xlabel, figname = callback(df_temp, method)
            # there is an error if nothing is returned
            if df is None:
                logger.error("No df returned from byte datetime groupers.\n\
                             Method = "+method)
                return
            x, y = getSortedCDF(list(df))
            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % df.mean()
                lab += ': '+ calcMean
            ax1[next(axis_ctr)].plot(x,y, color=next(colors), label=lab,
                                     marker=markers.next(), markevery=len(y)/10)

        ax1[0].set_xscale(self.XSCALE)
        ax1[1].set_xscale(self.XSCALE)

        ax1[1].set_xlabel(xlabel)
        ax1[0].set_ylabel("Normalized CDF UPLINK")
        ax1[1].set_ylabel("Normalized CDF DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(self.OUTPUTPATH+figname)
        return

##############################################################################
# MAIN methods
##############################################################################

DATASET = 'test-data-set.dat'
CONTROLSET = 'test-control-set.dat'
DATALABEL = 'test-data-set'
LIST_OF_CONTROLLABELS = ['test-control-set']
NDEVICES = 1000

def plotByDateRange(ndev=NDEVICES, DataSet=DATALABEL, ControlSets=LIST_OF_CONTROLLABELS):
    """
    load test set up and down
    for control set in [control1 - control8]
    load control set up and down and get date_start, date_end
    slice control set to 1000 devices,
    slice test set to 1000 devices, date_start, date_end
    plot CDFs
    """
    # Plotter Class Object
    p = Plotter()
    # load test set up and down
    DATA_up, DATA_dw = load_dataframe(DataSet)
    logger.debug("loaded test sets "+ DataSet )
    # for control set in [control1 - control8]
    for ControlSet in ControlSets:
        CONTROL_up, CONTROL_dw = load_dataframe(ControlSet)
        logger.debug("loaded control sets "+ ControlSet )

        # get date_start, date_end
        date_start, date_stop = get_dates(CONTROL_up)
        # slice
        CUP_sliced = slice_df(CONTROL_up, ndev, date_start, date_stop)
        DUP_sliced = slice_df(DATA_up, ndev, date_start, date_stop)
        logger.debug("sliced UP sets ")

        # get date_start, date_end
        date_start, date_stop = get_dates(CONTROL_dw)
        # slice
        CDW_sliced = slice_df(CONTROL_dw, ndev, date_start, date_stop)
        DDW_sliced = slice_df(DATA_dw, ndev, date_start, date_stop)
        logger.debug("sliced DW sets ")

        # plot CDFs
        # new outputpath by control-set and date
        outputlabel = ControlSet + '_' + date_start.replace(" ","_") \
        + '_' + date_stop.replace(" ","_")
        logger.info("Plots stored in " + outputlabel)
        p.OUTPUTPATH = 'output/' + outputlabel + '/'
        if not os.path.exists(p.OUTPUTPATH):
            os.makedirs(p.OUTPUTPATH)
        # load test and control up and dw sets
        p.import_df(DUP_sliced, DDW_sliced, CUP_sliced, CDW_sliced)
        p.DLABEL = DataSet
        p.CLABEL = ControlSet

        # allBytes and nonZeroBytes
        p.CDFPlotter(p.allBytes, '')
        p.CDFPlotter(p.thresholdBytes, 0)
        # max, median, 90 percentile
        for method in ['Peak', 'Median', '90perc']:
            p.CDFPlotter(p.getBytesPerDevice, method)
            logger.debug("Bytes per Device method = " + method)
            p.CDFPlotter(p.getBytesPerDevicePerDay, method)
            logger.debug("Bytes per Device per Day method = " + method)
            p.CDFPlotter(p.getBytesPerDevicePerTimeSlot, method)
            logger.debug("Bytes per Device per Time method = " + method)

        logger.debug("DONE control: " + ControlSet + "; test: " + DataSet)

    return DUP_sliced, DDW_sliced, CUP_sliced, CDW_sliced

def main():
    p = Plotter()
    p.load_df()
    p.splitDateTime()
    p.CDFPlotter(p.allBytes)
    p.CDFPlotter(p.nonZeroBytes)
    p.CDFPlotter(p.peakBytesPerDevice)
    p.CDFPlotter(p.peakBytesPerDevicePerDay)
    p.CDFPlotter(p.peakBytesPerDevicePerTimeSlot)
    return
