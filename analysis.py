import pandas as pd
import csv
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime
import pickle as pkl
import os
import numpy as np
import logging

#PROCESSEDPATH='processed/'
DATAPATH='/data/users/sarthak/comcast-data/'
PROCESSEDPATH='/data/users/sarthak/comcast-data/processed/'
OUTPUTPATH='output/'

'''
Generate a test data/control set for quick play
1. test-data-set
head -300 250-test.dat > test-data-set.dat
tail -300 250-test.dat >> test-data-set.dat
2. test-control-set
head -300 control1.dat > test-control-set.dat
tail -300 control1.dat >> test-control-set.dat
'''

DATASET='test-data-set' #'250-test'
CONTROLSET='test-control-set'#'control1'

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
DEBUG_ROW_DISPLAY = 100 #100000      #display every 100,000th row in original data
DEBUG_LOG_COUNTER_DISPLAY = 1 #100     #display every 100th device number

# DATA SPLITTING

def dat2csv_by_deviceNum(datasetname):
    """
    separate huge csv into multiple deviceNumber.csv
    also store peakBytesPerUser
    """
    folder = datasetname.split('.')[0]
    if not os.path.exists(PROCESSEDPATH + folder):
        os.makedirs(PROCESSEDPATH + folder)

    with open(DATAPATH + datasetname, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        count = 0
        for row in csvreader:
            deviceNum = row[0]
            # append data to avoid loading huge arrays
            f = open(PROCESSEDPATH + folder + '/' + deviceNum + '.csv', 'a')
            writer = csv.writer(f)
            # write to pickle by Device_numbers
            # Device_number, end_time, cmts_inet, service_direction,
            # octets_passed
            writer.writerow( (row[0], row[1], row[4], row[5], row[7]) )
            f.close()
            #print every 100,000th row in the dataset
            count += 1
            if count % DEBUG_ROW_DISPLAY == 0:
                logger.debug(datasetname + ' ' + row[0] + ' ' + row[1])
        logger.info("Done converting to individual csv")
    return

def create_csv(datasetname):
    dat2csv_by_deviceNum(datasetname)
    return

def read_dataframe(folder, filename):
    header_row = ['device_number', 'end_time', 'cmts_inet', 'direction', 'bytes']
    return pd.read_csv(PROCESSEDPATH + folder + '/' + filename, names=header_row)

# Calculate peak usage per day for each device num
# Calculate non zero usage per day for each device num

# GLOBAL lists to get CDFs
noZeroHist = {'up':Counter([]), 'dw':Counter([])}
peakHist = {'up':Counter([]), 'dw':Counter([])}
peakBytesPerUser = {'up':defaultdict(int), 'dw':defaultdict(int)}

def update_cdf(filename, folder):

    df_dev = pd.read_csv(PROCESSEDPATH + folder + '/' + filename, names = header_row)
    # split datetime to date and time to make grouping easier
    df_dev['date'] = df_dev['end_time'].apply(lambda x: x.split(" ")[0])
    df_dev['time'] = df_dev['end_time'].apply(lambda x: x.split(" ")[1])

    # split into dw (1) and up (2)
    df_dw = df_dev[df_dev['service_direction'] == 1]
    df_up = df_dev[df_dev['service_direction'] == 2]

    # 0. Peak bytes per day per user
    # get max of the file upstream or downstream
    deviceNum = int(filename.rstrip('.csv'))
    peakBytesPerUser['up'][deviceNum] = df_up['octets_passed'].max()
    peakBytesPerUser['dw'][deviceNum] = df_dw['octets_passed'].max()
    logger.debug('peakByesPerUser["up"] = '+str(peakBytesPerUser['up']))

    # 1. Peak Per Day CDF
    # group by direction and date; find indices for peak octets_passed per day
    # transform(max) returns the actual max value per group for each index
    # compare with original table to get only indices for which max was true
    idx1 = df_dw.groupby(["date"], sort=False)['octets_passed'].transform(max) == df_dw['octets_passed']
    idx2 = df_up.groupby(["date"], sort=False)['octets_passed'].transform(max) == df_up['octets_passed']

    # unfortunately there may be duplicates for which max per groupby matched - just select one instead
    peakPerDay_dw = df_dw[idx1].groupby("date").first()
    peakPerDay_up = df_up[idx2].groupby("date").first()

    # save in huge list over which CDF or hist will be plotted
    #peakHist['dw'].extend( list(peakPerDay_dw['octets_passed']) )
    #peakHist['up'].extend( list(peakPerDay_up['octets_passed']) )
    peakHist['dw'] += Counter( list(peakPerDay_dw['octets_passed']) )
    peakHist['up'] += Counter( list(peakPerDay_up['octets_passed']) )

    # 2. Drop Zeros
    dropZeros = df_dev[df_dev['octets_passed'] != 0]
    noZeroHist['dw'] += Counter( list(dropZeros[dropZeros['service_direction'] == 1]['octets_passed']) )
    noZeroHist['up'] += Counter( list(dropZeros[dropZeros['service_direction'] == 2]['octets_passed']) )
    #noZeroHist['dw'].extend( list(dropZeros[dropZeros['service_direction'] == 1]['octets_passed']) )
    #noZeroHist['up'].extend( list(dropZeros[dropZeros['service_direction'] == 2]['octets_passed']) )
    return


# Save byte lists (nonZero and peak) for control and data set
def save_hists(folder):
    log_counter = 0

    for filename in os.listdir(PROCESSEDPATH + folder):
        # display every filename after 100 counts (almost 22 times)
        log_counter += 1
        update_cdf(filename, folder)
        #if (int(filename.split('.')[0]) % 100 == 0): logger.debug("Counter " + str(log_counter) + ": " + filename)
        if (log_counter % DEBUG_LOG_COUNTER_DISPLAY == 0): logger.debug("Counter " + str(log_counter) + ": " + filename)

    logger.info("Done updating CDF variables. Now save hists in pkl")
    # save peakBytes per User pkl
    folder = datasetname.split('.')[0]
    if not os.path.exists(OUTPUTPATH+folder):
        os.makedirs(OUTPUTPATH+folder+"/")
    f = open(OUTPUTPATH+folder+"/"+'peakBytesPerUser.pkl', 'w')
    pkl.dump(peakBytesPerUser, f)
    f.close()
    logger.info("Pickled peakBytesPerUser for " + folder)
    # pickle hist series data
    pkl.dump(peakHist, open(OUTPUTPATH+folder+"/"+'peakCounter.pkl', 'w'))
    logger.info("Pickled peakHist for " + folder)
    pkl.dump(noZeroHist, open(OUTPUTPATH+folder+"/"+'noZeroCounter.pkl', 'w'))
    logger.info("Pickled noZeroHist for " + folder)
    return


if __name__ == "__main__":

    CREATE_CSV = 1
    ANALYZE = 1

    #for datasetname in ['test-data-set.dat', 'test-control-set.dat']:
    datasets = [x+'.dat' for x in [DATASET, CONTROLSET]]

    for datasetname in datasets:
        folder = datasetname.split('.')[0]
        if (CREATE_CSV==1):
            create_csv(datasetname)

        if (ANALYZE==1):
            # For each data set
            noZeroHist = {'up':Counter([]), 'dw':Counter([])}
            peakHist = {'up':Counter([]), 'dw':Counter([])}
            peakBytesPerUser = {'up':defaultdict(int), 'dw':defaultdict(int)}
            log_counter = 0

            # save pkls nonzero and peak
            save_hists(folder)
