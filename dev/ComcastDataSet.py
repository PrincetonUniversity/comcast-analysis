from __future__ import division
import pandas as pd
import numpy as np
import os, random
import struct, socket
import pickle as pkl
from collections import defaultdict
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

CONST = {}
CONST['DATAPATH'] = '/data/users/sarthak/comcast-data/data/'
CONST['FILTEREDPATH'] = '/data/users/sarthak/comcast-data/filtered/'
CONST['PROCESSEDPATH'] = '/data/users/sarthak/comcast-data/processed/'
CONST['OUTPUTPATH'] = '/data/users/sarthak/comcast-data/output/'
CONST['SANITIZEDPATH'] = '/data/users/sarthak/comcast-data/sanitized'
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
CONST['OCTETS_TO_Mbps'] = 8 / (15*60 * 1024 * 1024)

for pathname, path in CONST.iteritems():
    if 'PATH' in pathname and not os.path.exists(path):
        logger.info("Create path "+path+" for "+pathname)
        os.makedirs(path)

# original 250-test.dat size is 95,000,000 rows
# head -1000000 250-test.dat > test-data-set.dat
# tail -1000000 250-test.dat >> test-data-set.dat
# head -1000000 control1.dat > test-control-set.dat
# tail -1000000 control1.dat >> test-control-set.dat
# when running in /data/users/sarthak it takes less than a min to load full

#############################################################################
# CLASS DEFINITION
#############################################################################

class ComcastDataSet:
    def __init__(self, **kwargs):
        self.NDEV = 1000
        self.name = None
        if 'FILENAME' in kwargs.keys():
            self._set_filename(kwargs['FILENAME'])
        else:
            self.datafilename = none
        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        self.data = pd.DataFrame({})
        self.up = pd.DataFrame({})
        self.dw = pd.DataFrame({})
        self.length = len(self.data)
        if 'header_row' in kwargs.keys():
            self.header_row = kwargs['header_row']
            self.filter_header_row = self.header_row
        else:
            self.header_row = 0
            self.filter_header_row = 0
        self._load_paths(**kwargs)

    def _set_filename(filename):
        self.datafilename = filename
        self._get_default_name_from_filename()
        return

    def _get_default_name_from_filename(self):
        self.name = self.datafilename.split('.')[0]
        logger.warning("Set name to " + self.name)
        return

    def _load_paths(self, **kwargs):
        self.DATAPATH = kwargs['DATAPATH']
        self.PROCESSEDPATH = kwargs['PROCESSEDPATH']
        self.OUTPUTPATH = kwargs['OUTPUTPATH']
        self.FILTEREDPATH = kwargs['FILTEREDPATH']
        self.SANITIZEDPATH = kwargs['SANITIZEDPATH']
        return

    # Load raw csv
    def _load_csv(self, datfile=None):
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
        self.filter_header_row = ['Device_number', 'end_time', 'cmts_inet'
                                  'octets_passed', 'service_class_name']
        df_up = df[df['service_direction'] == 2][self.filter_header_row]
        df_dw = df[df['service_direction'] == 1][self.filter_header_row]
        df_up.to_pickle(self.PROCESSEDPATH + self.name+'_up.pkl')
        df_dw.to_pickle(self.PROCESSEDPATH + self.name+'_dw.pkl')
        self.up = df_up
        self.dw = df_dw
        return

    # load direction split datasets
    def main_load_full(self):
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

    # PROCESS: aggregate service_class_name, add extra cols, and save

    def main_load_processed(self):
        name = self.name
        self.up = pd.read_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw = pd.read_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    def main_aggregate(self):
        """
        groupby service_class_name and add all octets
        """
        up_g = self.up.groupby(['Device_number', 'end_time', 'cmts_inet'], as_index=False)
        dw_g = self.dw.groupby(['Device_number', 'end_time', 'cmts_inet'], as_index=False)
        # sum the octets_passed column
        self.up = up_g.sum()
        self.dw = dw_g.sum()
        # can also use grouped.aggregate(np.sum)
        return

    def main_add_extra_columns(self):
        logger.warning("add datetime, throughput, ipaddr to up and dw")
        self._add_datetime()
        self._add_throughput()
        self._add_ipaddr()
        self.filter_header_row = ['Device_number', 'end_time', 'octets_passed',
                                  'datetime', 'throughput','ipaddr']
        #self.add_time()
        return

    def main_save_processed(self):
        name = self.name
        logger.warning("save_processed with extra cols: "+self.FILTEREDPATH + name+'_[direction].pkl')
        self.up[self.filter_header_row].to_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw[self.filter_header_row].to_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    # helper funcs to add columns
    def _add_datetime(self):
        logger.warning("add_datetime to up and dw using end_time")
        #self.up['time'] = self.up['end_time'].apply(lambda x: x.split()[1])
        self.up['datetime'] = pd.to_datetime(self.up['end_time'])
        #self.dw['time'] = self.dw['end_time'].apply(lambda x: x.split()[1])
        self.dw['datetime'] = pd.to_datetime(self.dw['end_time'])
        return

    def _add_time(self):
        """ not used """
        logger.warning("add_time to up and dw splitting end_time")
        self.up['time'] = self.up['end_time'].apply(lambda x: x.split()[1])
        self.dw['time'] = self.dw['end_time'].apply(lambda x: x.split()[1])
        self.filter_header_row.append('time')
        return

    def _add_throughput(self):
        logger.warning("add_throughput to up and dw using octets_passed")
        self.up['throughput'] = self.up.octets_passed * CONST['OCTETS_TO_Mbps']
        self.dw['throughput'] = self.dw.octets_passed * CONST['OCTETS_TO_Mbps']
        return

    def _add_ipaddr(self):
        logger.warning("add_ipaddr to up arnd dw using cmts_inet")
        get_ipaddr = lambda x: socket.inet_ntoa(struct.pack('!L', int( x )))
        self.up['ipaddr'] = self.up.cmts_inet.apply(get_ipaddr)
        self.dw['ipaddr'] = self.dw.cmts_inet.apply(get_ipaddr)
        return

    def _get_date_range(self):
        self.up.sort('end_time', ascending=True)
        date_start = self.up['end_time'].iloc[0]
        date_end = self.up['end_time'].iloc[-1]
        return date_start, date_end

    # Slice, Filter, choose direction datasets with extracols
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
            self.add_time()
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

    def save_ndev_filtered(self):
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

    # Save/Load filtered
    def main_save_filtered(self, attrib=None):
        name = self.name
        if not attrib is None:
            name += '-'+attrib
        logger.warning("save_filtered by filtered_header_row: "+self.FILTEREDPATH + name+'_[direction].pkl')
        self.up[self.filter_header_row].to_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw[self.filter_header_row].to_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    def main_load_filtered(self, attrib=None):
        name = self.name
        if not attrib is None:
            name += attrib
        self.up = pd.read_pickle(self.FILTEREDPATH + name+'_up.pkl')
        self.dw = pd.read_pickle(self.FILTEREDPATH + name+'_dw.pkl')
        return

    # MAIN: FINAL aggregation and save after device, daterange
    def load(self, attrib=None):
        """main method to load a df and save it"""
        name = self.name
        if not attrib is None:
            name += attrib
        if not os.path.isfile (self.FILTEREDPATH + name+'_up.pkl'):
            logger.warning("Could not find: "+self.FILTEREDPATH + name+'_up.pkl')
            self.main_load_full()
            logger.debug("no filtering, aggregate and add extra columns")
            self.main_aggregate()
            self.main_add_extra_columns()
            self.main_save_processed()
        else:
            self.main_load_processed()
        return

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
