from __future__ import division
import pandas as pd
import numpy as np
import random, os
import matplotlib
import pickle as pkl
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

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

#LIST_OF_CONTROLLABELS = ['control5']
#LIST_OF_CONTROLLABELS = ['control'+str(l) for l in range(1,9)]
#LIST_OF_CONTROLLABELS.remove('control3')         #causes errors unslicable

foldername = CONTROLLABEL
DATAPATH = '../data/'
FILTEREDPATH = '../filtered/'
PROCESSEDPATH = '../processed/' + foldername + '/'
OUTPUTPATH = '../output/' + foldername + '/'
CONST = {}
CONST['DATAPATH'] = '../data/'
CONST['FILTEREDPATH'] = '../filtered/'
CONST['PROCESSEDPATH'] = '../processed/' + foldername + '/'
CONST['OUTPUTPATH'] = '../output/' + foldername + '/'
for pathname, path in CONST.iteritems():
    if not os.path.exists(path):
        logger.info("Creat path "+path+" for "+pathname)
        os.makedirs(path)

CONST['header_row'] = ['Device_number',
         'end_time',
         'date_service_created',
         'service_class_name',
         'cmts_inet',
         'service_direction',
         'port_name',
         'octets_passed',
         'device_key',
         'service_identifier']
#############################################################################
# load dataframes
#############################################################################

class ComcastDataSet:
    def __init__(self, **kwargs):
        self.DATAPATH = kwargs['DATAPATH']
        self.PROCESSEDPATH = kwargs['PROCESSEDPATH']
        self.OUTPUTPATH = kwargs['OUTPUTPATH']
        self.FILTEREDPATH = kwargs['FILTEREDPATH']
        self.NDEV = 1000
        self.name = None
        if 'FILENAME' in kwargs.keys():
            self.datafilename = kwargs['FILENAME']
            self._get_default_name_from_filename()
        else:
            self.datafilename = None
        if 'NAME' in kwargs.keys():
            self.name = kwargs['NAME']
        self.data = pd.DataFrame({})
        self.up = 0
        self.dw = 0
        self.length = len(self.data)
        if 'header_row' in kwargs.keys():
            self.header_row = kwargs['header_row']
            self.filter_header_row = self.header_row
        else:
            self.header_row = 0
            self.filter_header_row = 0

    def load_paths(self, **kwargs):
        self.DATAPATH = kwargs['DATAPATH']
        self.PROCESSEDPATH = kwargs['PROCESSEDPATH']
        self.OUTPUTPATH = kwargs['OUTPUTPATH']
        self.FILTEREDPATH = kwargs['FILTEREDPATH']

    def _get_default_name_from_filename(self):
        self.name = self.datafilename.split('.')[0]
        logger.warning("Set name to " + self.name)
        return

    # Load full
    def load_csv(self, datfile=None):
        if datfile is None:
            if self.datafilename is None:
                logger.error("Using "+self.name+".dat as filename")
                self.datafilename = self.name+'.dat'
        else:
            self.datafilename = datfile

        df = pd.read_csv(self.DATAPATH + self.datafilename, delimiter='|', names=self.header_row)
        df.sort('end_time', ascending=True)
        self._get_default_name_from_filename()
        self.data = df
        return

    # Split by direction and save
    def save_by_direction(self):
        logger.warning("save_by_direction: "+self.PROCESSEDPATH + self.name+'_up.pkl')
        df = self.data
        self.filter_header_row = ['Device_number', 'end_time',
                                  'octets_passed', 'service_class_name']
        df_up = df[df['service_direction'] == 2][self.filter_header_row]
        df_dw = df[df['service_direction'] == 1][self.filter_header_row]
        df_up.to_pickle(self.PROCESSEDPATH + self.name+'_up.pkl')
        df_dw.to_pickle(self.PROCESSEDPATH + self.name+'_dw.pkl')
        self.up = df_up
        self.dw = df_dw
        return

    def load_full(self):
        if not os.path.isfile (self.PROCESSEDPATH + self.name + '_up.pkl'):
            logger.warning("Could not find: "+self.PROCESSEDPATH + self.name+'_up.pkl')
            self.load_csv(self.datafilename)
            self.save_by_direction()
        else:
            self.up = pd.read_pickle(self.PROCESSEDPATH + self.name+'_up.pkl')
            self.dw = pd.read_pickle(self.PROCESSEDPATH + self.name+'_dw.pkl')
            self.length = len(self.up) + len(self.dw)
            self.filter_header_row = list(self.up.columns)
        return

    # Filter and save
    def add_datetime(self):
        logger.warning("add_datetime: split up and dw end_time to datetime and time")
        self.up['time'] = self.up['end_time'].apply(lambda x: x.split()[1])
        self.up['datetime'] = pd.to_datetime(self.up['end_time'])
        self.dw['time'] = self.dw['end_time'].apply(lambda x: x.split()[1])
        self.dw['datetime'] = pd.to_datetime(self.dw['end_time'])
        self.filter_header_row = list(self.up.columns)
        return

    def get_date_range(self):
        self.up.sort('end_time', ascending=True)
        date_start = self.up['end_time'].iloc[0]
        date_end = self.up['end_time'].iloc[-1]
        return date_start, date_end

    def save_filtered(self, attrib=None):
        name = self.name
        if not attrib is None:
            name += '-'+attrib
        logger.warning("save_filtered: "+self.FILTEREDPATH + name+'_[direction].pkl')
        self.up.to_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw.to_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    def load_filtered(self, attrib=None):
        name = self.name
        if not attrib is None:
            name += attrib
        self.up = pd.read_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw = pd.read_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    # Slice and choose devices
    def choose_devices(self, ndev):
        device_file = self.FILTEREDPATH + self.name + str(ndev) + '.pkl'
        if os.path.isfile(device_file):
            devices = pkl.load(open(device_file, 'r'))
        else:
            logger.warning("Could not find: "+device_file+". Choose "+str(ndev)+" and save it.")
            devices = random.sample(list(self.up['Device_number'].unique()), ndev)
            pkl.dump(devices, open(device_file, 'w'))
        return devices

    def filter_by_device(self, devices=None):
        if devices is None:
            devices = self.choose_devices(self.NDEV)

        self.up = self.up [ self.up ['Device_number'].isin(devices) ]
        self.dw = self.dw [ self.dw ['Device_number'].isin(devices) ]
        return

    def filter_by_date(self, date_start, date_stop):
        self.up =  self.up [ (self.up['end_time'] >= date_start) & (self.up['end_time'] <= date_stop) ]
        self.dw =  self.dw [ (self.dw['end_time'] >= date_start) & (self.dw['end_time'] <= date_stop) ]
        return

    def filter_by_time(self, times):
        if not 'time' in list(self.up.columns):
            self.add_datetime()
        self.up = self.up[self.up['time'].isin(times)]
        self.dw = self.dw[self.dw['time'].isin(times)]
        return

    def filter_by_service(self, keywords):
        self.up = self.up[ self.up['service_class_name'].isin(keywords) ]
        self.dw = self.dw[ self.dw['service_class_name'].isin(keywords) ]
        return

    def filter_agg_service(self, aggType='sum'):
        for df in [self.up, self.dw]:
            #TODO incomplete
            df = df.groupby([ list(df.columns).remove('service_class_name') ], as_index=False).sum()
        return

    def filter_bytes_threshold(self, threshold):
        #TODO
        for df in [self.up, self.dw]:
            df = df[ df['octets_passed'] > threshold]
        return

    def load(self, attrib=None):
        """main method to load a df and save it"""
        name = self.name
        if not attrib is None:
            name += attrib
        if not os.path.isfile (self.FILTEREDPATH + name+'_up.pkl'):
            logger.warning("Could not find: "+self.FILTEREDPATH + name+'_up.pkl')
            self.load_full()
            self.save_filtered(attrib)
        else:
            self.load_filtered(attrib)
        return

    # FINAL aggregation and save after device, daterange
    def save_processed(self):
        filepath = self.PROCESSEDPATH + self.name+'_[dir]-'+str(self.NDEV)+'.pkl'
        logger.debug("Save filterd and chosen test/control set to: "+filepath)
        self.up.to_pickle(self.PROCESSEDPATH + self.name+'_up-'+str(self.NDEV)+'.pkl')
        self.dw.to_pickle(self.PROCESSEDPATH + self.name+'_dw-'+str(self.NDEV)+'.pkl')
        return

    def aggregate(self, method='sum'):
        """
        current df contains too many columns. Use [sum|max|wifi]
        to reduce data into the form [device_id, end_time]:octets_passed
        save these datasets to filtered as [method]-[name]_[dir].pkl
        """
        up_g = self.up.groupby(['Device_number', 'end_time'], as_index=False)
        dw_g = self.dw.groupby(['Device_number', 'end_time'], as_index=False)
        if method=='max':
            self.up = up_g.max()
            self.dw = dw_g.max()
        else:
            method = 'sum'
            self.up = up_g.sum()
            self.dw = dw_g.sum()
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

def slicer(ControlSet, DATA_up, DATA_dw, ndev):
    CONTROL_up, CONTROL_dw = load_dataframe(ControlSet)
    # get date_start, date_end
    date_start, date_stop = get_dates(CONTROL_up)
    # slice
    logger.debug(ControlSet + " ndev=" + str(ndev) + " control set unique: "\
        + str(len( CONTROL_up['Device_number'].unique() )) + " data set unique: "\
        + str(len( DATA_up['Device_number'].unique() )) )

    CUP_sliced = slice_df(CONTROL_up, ndev, date_start, date_stop)
    DUP_sliced = slice_df(DATA_up, ndev, date_start, date_stop)
    #logger.debug("sliced UP sets ")

    # get date_start, date_end
    date_start, date_stop = get_dates(CONTROL_dw)
    # slice
    logger.debug(ControlSet + " ndev=" + str(ndev) + " control set unique: "\
        + str(len( CONTROL_dw['Device_number'].unique() )) + " data set unique: "\
        + str(len( DATA_dw['Device_number'].unique() )) )
    CDW_sliced = slice_df(CONTROL_dw, ndev, date_start, date_stop)
    DDW_sliced = slice_df(DATA_dw, ndev, date_start, date_stop)

    # return name of outputfolder
    outputlabel = ControlSet + '_' + date_start.replace(" ","_") \
        + '_' + date_stop.replace(" ","_")
    #logger.debug("sliced DW sets ")
    return DUP_sliced, DDW_sliced, CUP_sliced, CDW_sliced, outputlabel


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
        self.ALL_DF = {'test_up': self.test_up,
                       'test_dw': self.test_dw,
                       'control_up': self.control_up,
                       'control_dw': self.control_dw}
        self.grouped = defaultdict(int)
        return

    def allBytes(self, df):
        return df

    def nonZeroBytes(self, df):
        return df[df['octets_passed'] > 0]

    def thresholdBytes(self, df, threshold=0):
        """
        return df, xlabel, filename
        """

        dfr = df[df['octets_passed'] > threshold]
        return dfr, "Bytes > "+str(threshold),"CDF-ThresholdBytes-"+str(threshold)

    def getStringsFromGroupBy(self, GROUPBY):
        """
        eg: input = ["Device_number", "time"]
        returns:
            xlab = " Bytes per Device-number per Time"
            filename = "Bytes_Device-number_Time"
        """

        xlab = " Bytes "
        filename = "Bytes"
        for col in GROUPBY:
            col2 = col.capitalize().replace("_", "-")
            xlab += " per "+col2
            filename += "_"+col2
        logger.debug("xlab, filename = "+xlab+"; "+filename)
        return xlab, filename

    def grouper(self, GROUPBY):
        """
        save df_grouped['octets_passed'] by ['Device_number'],
        ['Device_number', 'date'], ['Device_number', 'time'], ['end_time']
        """
        logger.debug("group ALL_DF according to "+str(GROUPBY))
        for name, df in self.ALL_DF.items():
            self.grouped[name] = df.groupby(GROUPBY, as_index=False)['octets_passed']
            logger.debug( "Group "+name+" by "+str(GROUPBY)+" len: "+str( len(self.grouped[name].first()) ) )
        return

    def getByteSeries(self, method, **kwargs):
        """
        for each sliced set
        method = all: series = df['octets_passed']
        method = threshold: series = thresholdBytes['octets_passed']
        method = peak, group = device: series = getGroupedByteStats(grouped).max()
        """

        self.byte_series = defaultdict(int)

        logger.debug("enter getByteSeries; method = "+method)

        if method=='all':
            #self.byte_series = self.ALL_DF
            for name, df in self.ALL_DF.items():
                self.byte_series[name] = self.allBytes(df)['octets_passed']
        elif method=='nonZero':
            for name, df in self.ALL_DF.items():
                self.byte_series[name] = self.nonZeroBytes(df)['octets_passed']
        elif method=='threshold':
            for name, df in self.ALL_DF.items():
                self.byte_series[name] = self.thresholdBytes(df, kwargs['threshold'])['octets_passed']
        elif method=='Peak':
            for name, gr in self.grouped.items():
                self.byte_series[name] = gr.max()['octets_passed']
                #logger.debug("name = "+name+"; max series len = "+ str(len(self.byte_series[name])))
        elif method=='Median':
            for name, gr in self.grouped.items():
                self.byte_series[name] = gr.median()['octets_passed']
        elif method=='90perc':
            for name, gr in self.grouped.items():
                #self.byte_series[name] = gr.quantile(.90)['octets_passed']
                self.byte_series[name] = gr.apply(lambda x: np.percentile(x, 90))
        else:
            logger.error("Unknown method = "+method)
        return

    def CDFPlotter(self, xlabel, figname):
        """
        given self BYTES[ ]: test_up/dw control_up/dw
        just plot in CDF
        """
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))

        labels = iter([self.DLABEL+'_up',self.CLABEL+'_up',self.DLABEL+'_dw',
                       self.CLABEL+'_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])
        markers = iter(['o','x','o','x'])

        #logger.debug("byte_series.items() " + str(self.byte_series.items()))
        for name, series in self.byte_series.items():
            x, y = getSortedCDF(list(series))

            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % np.mean(list(series))
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

    def setOutputPath(self, outputlabel):
        # new outputpath by control-set and date
        logger.info("Plots stored in " + outputlabel)
        self.OUTPUTPATH = 'output/' + outputlabel + '/'
        if not os.path.exists(self.OUTPUTPATH):
            logger.debug("MakeDir " + self.OUTPUTPATH)
            os.makedirs(self.OUTPUTPATH)
        return


    def SeriesPlotter(self, ylabel, figname):
        """
        given self BYTES[ ]: test_up/dw control_up/dw
        just plot in CDF
        """
        logger.debug("enter seriesplotter")
        fig1, ax1 = plt.subplots(2,1, figsize=(18,10))

        labels = iter([self.DLABEL+'_up',self.CLABEL+'_up',self.DLABEL+'_dw',
                       self.CLABEL+'_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])
        markers = iter(['o','d','o','d'])

        #logger.debug("byte_series.items() " + str(self.byte_series.items()))
        for name, series in self.byte_series.items():
            x = pd.to_datetime(series.index)
            y = series.values

            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % np.mean(list(series))
                lab += ': '+ calcMean
            ax1[next(axis_ctr)].plot_date(x,y, color=next(colors), label=lab,
                                          alpha=0.5, marker=markers.next())#,
                                     #marker=markers.next(), markevery=len(y)/10)

        #ax1[0].set_xscale(self.XSCALE)
        #ax1[1].set_xscale(self.XSCALE)

        ax1[1].set_xlabel("DateTime")
        ax1[0].set_ylabel(ylabel + " Bytes UPLINK")
        ax1[1].set_ylabel(ylabel + " Byes DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(self.OUTPUTPATH+figname)

        return

    def plotAll(self):
        """
        main method
        - Has sliced self.test_up, self.test_dw, self.control_up, self.control_dw
        - Plot
        # allBytes and nonZero then CDF all stats
        # group by dev [dev, dev-day, dev-time] then CDF all stats
        # group by [datetime] then CDF all stats
        # group by [datetime] then timeseries plot (x, y)
        """

        # allBytes and nonZero then CDF all stats
        self.getByteSeries('all')
        xlabel = 'All Bytes Seen'
        figname = 'CDF-allBytes'
        self.CDFPlotter(xlabel, figname)
        logger.debug("draw all CDF")

        self.getByteSeries('nonZero')
        xlabel = 'Bytes > 0'
        figname = 'CDF-nonZeroBytes'
        self.CDFPlotter(xlabel, figname)
        logger.debug("draw nonZero CDF")

        # group by dev [dev, dev-day, dev-time] then CDF all stats
        # group by [datetime] then CDF all stats
        for group_by in [['Device_number'], ['Device_number', 'date'],
                       ['Device_number', 'time'], ['end_time']]:
            # store each group in self.grouped
            self.grouper(group_by)
            xlab, filename = self.getStringsFromGroupBy(group_by)
            for stats in ['Peak', 'Median', '90perc']:
                self.getByteSeries(stats)
                xlabel = stats + xlab
                figname = 'CDF-'+filename+'-'+stats
                self.CDFPlotter(xlabel, figname)
                logger.debug("draw CDF - Bytes per "+str(group_by)+" "+stats)

        # for the last groupby ['end_time']
        # group by [datetime] then timeseries plot (x, y)
        GROUPBY = ['end_time']
        xlab, filename = self.getStringsFromGroupBy(GROUPBY)
        for name, df in self.ALL_DF.items():
            self.grouped[name] = df.groupby(GROUPBY)

        for stats in ['Peak', 'Median', '90perc']:
            #return self.grouped
            if stats=='Peak':
                for name, df in self.grouped.items():
                    self.byte_series[name] = df['octets_passed'].apply(lambda x: max(x))
            elif stats=='Median':
                for name, df in self.grouped.items():
                    self.byte_series[name] = df['octets_passed'].apply(lambda x: np.median(x))
            elif stats=='90perc':
                for name, df in self.grouped.items():
                    self.byte_series[name] = df['octets_passed'].apply(lambda x: np.percentile(x, 90))
            else:
                logging.error("invalid stat, self.byte_series may be corrupted")
            #self.getByteSeries(stats)
            ylabel = stats + xlab
            figname = 'TimeSeries-'+filename+'-'+stats
            self.SeriesPlotter(ylabel, figname)

        return

##############################################################################
# MAIN methods
##############################################################################

#DATASET = 'test-data-set.dat'
#CONTROLSET = 'test-control-set.dat'
#DATALABEL = 'test-data-set'
LIST_OF_CONTROLLABELS = ['test-control-set']
NDEVICES = 1000

def main2(ndev=NDEVICES, DataSet=DATALABEL, ControlSets=LIST_OF_CONTROLLABELS, method='sum'):
    logger.info("NDEVICES "+str(ndev)+" DateSet "+DataSet+" ControlSets "+str(ControlSets))
    # Plotter Class Object
    p = Plotter()
    # load test set up and down
    DATA_up, DATA_dw = load_dataframe(DataSet)
    # for control set in [control1 - control8]
    for ControlSet in ControlSets:
        # load test and control up and dw sets
        test_up, test_dw, control_up, control_dw, outputlabel = slicer(ControlSet, DATA_up,
                                                          DATA_dw, ndev)

        test_up = process_df(test_up, method)
        test_dw = process_df(test_up, method)
        control_up = process_df(control_up, method)
        control_dw = process_df(control_up, method)
        outputlabel = method.upper() + '/' + outputlabel

        p.setOutputPath(outputlabel)
        p.import_df(test_up, test_dw, control_up, control_dw)
        p.DLABEL = DataSet
        p.CLABEL = ControlSet

        x = p.plotAll()
        logger.info("Done "+ControlSet)

    return p, x

def process_df(df, method='sum'):
    g = df.groupby(['Device_number', 'end_time', 'date', 'time'], as_index=False)
    if method=='sum':
        return g.sum()
    elif method=='max':
        return g.max()
    elif method=='wifi':
        return df[df['service_class_name'] == 'xwifi_dn' | df['service_class_name'] == 'xwifi_up']
    else:
        return g.apply(lambda x: np.percentile(x, method))

