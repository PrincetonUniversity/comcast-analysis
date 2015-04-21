from __future__ import division
import pandas as pd
import os, sys, datetime
import numpy as np
from collections import defaultdict

'''
Aim: select only those control devices that exist throughout
the date range 30 sept to 29 dec
'''
DATE_START = '2014-09-30'
DATE_END = '2014-12-29'
THRESH = 0.8
# max avail threshold = 0.8 of full range
ts = pd.date_range(DATE_START, DATE_END, freq='15min')
HEARTBEAT_THRESH = THRESH * len(ts)

DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_dw"
#DATAFOLDER = "/data/users/sarthak/comcast-data/data/"
FINALFOLDER='/data/users/sarthak/comcast-data/separated/final_dw/'
if not os.path.exists(FINALFOLDER):
    os.makedirs(FINALFOLDER)

def try_sanitize():
# read input
    df2 = pd.read_pickle(DATAFOLDER+"control.pkl")

# find each devices start date and stop date
    dates_device_min = df2.groupby('Device_number')['octets_passed'].min()
    dates_device_max = df2.groupby('Device_number')['octets_passed'].max()
    dates_device = pd.DataFrame({'max': dates_device_max, 'min':dates_device_min})

# as these are ALREADY sanitized, there avail should be at least 0.8
# just check for first date and last date on the dataset
    dates_device['min_date'] = dates_device['min'].apply(lambda x: x.date())
    dates_device['max_date'] = dates_device['max'].apply(lambda x: x.date())

# get only devices with that date range
    all_range_date = dates_device[
        (dates_device['min_date']==dates_device['min_date'].min())
        & (dates_device['max_date']==dates_device['max_date'].max()) ]
    valid_control_devices = all_range_date.index

# save only valid devices
    df_control = df2[ df2['Device_number'].isin(valid_control_devices) ]
    df_control[['Device_number','octets_passed','datetime']].to_pickle(FINALFOLDER+"control.pkl")

# test set filter
    df1 = pd.read_pickle(DATAFOLDER+"test.pkl")
    df1[['Device_number','octets_passed','datetime']].to_pickle("../../final/test.pkl")
    return


'''
A better way: count num timeslots for both test and control
timeslots between 2014-09-30 and 2014-12-29 = x
timeslots_per_dev = df.groupby('Device_number')['datetime'].count()
valid = timeslots_per_dev [timeslots_per_dev > 0.8x]
final_control = df['Device_number'].isin(valid.index)
'''

def load_df(name, processed=1):
    if processed==1:
        DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_"+direction+"/"
        if name=='test':
            df1 = pd.read_pickle(DATAFOLDER+"test.pkl")
            return df1
        else:
            df2 = pd.read_pickle(DATAFOLDER+"control.pkl")
            return df2

    else:
        DATAFOLDER = "/data/users/sarthak/comcast-data/data/"
        header_row= ['Device_number', 'end_time', 'date_service_created',
                    'service_class_name', 'cmts_inet', 'service_direction',
                    'port_name', 'octets_passed', 'device_key',
                    'service_identifier']

        if name == 'test':
            if os.path.isfile(DATAFOLDER + "full_test.pkl"):
                return pd.read_pickle(DATAFOLDER+"full_test.pkl")
            datafilename = '250-test.dat'
            df1 = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time'], as_index=False).sum().reset_index()
            df1['name'] = datafilename
            print "Done test set load"
            df1['datetime'] = pd.to_datetime(df1['end_time'])
            # ADJUST DATETIME
            df1['datetime']-=datetime.timedelta(hours=6)
            df1.to_pickle(DATAFOLDER + "full_test.pkl")

            #direction = 'dw'
            #df_dw = df1[df1['service_direction']==1]
            #df_dw.to_pickle(DATAFOLDER+"full_test_"+direction+".pkl")

            #direction = 'up'
            #df_up = df1[df1['service_direction']==2]
            #df_up.to_pickle(DATAFOLDER+"full_test_"+direction+".pkl")
            return df1

        else:
            if os.path.isfile(DATAFOLDER + "full_control.pkl"):
                return pd.read_pickle(DATAFOLDER+"full_control.pkl")
            all_df = defaultdict(int)
            for datafilename in ['control'+str(x)+'.dat' for x in range(1,9)]:
                print "load " + datafilename
                df_temp = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time'], as_index=False).sum()
                all_df[ datafilename ] = df_temp
            df2 = pd.concat(all_df).reset_index().rename(columns={'level_0':'name'})
            print "Done control set load"
            df2['datetime'] = pd.to_datetime(df2['end_time'])
            # ADJUST DATETIME
            df2['datetime']-=datetime.timedelta(hours=6)
            df2.to_pickle(DATAFOLDER + "full_control.pkl")

            #direction = 'dw':
            #df_dw = df2[df2['service_direction']==1]
            #df_dw.to_pickle(DATAFOLDER+"full_control_"+direction+".pkl")

            #direction='up':
            #df_up = df2[df2['service_direction']==2]
            #df_up.to_pickle(DATAFOLDER+"full_control_"+direction+".pkl")
            return df2
    return

def add_extra_cols(df1, CONVERT_OCTETS= 8/(15 * 60 * 1024) ):
    """
    default convert from octets per 15 min to kbps
    """

    if not 'datetime' in df1.columns:
        df1['datetime'] = pd.to_datetime(df1['end_time'])
        # ADJUST DATETIME
        df1['datetime']-=datetime.timedelta(hours=6)

    df1['time'] = df1['datetime'].apply(lambda x: x.time())
    df1['date'] = df1['datetime'].apply(lambda x: x.date())
    df1['day'] = df1['datetime'].apply(lambda x: x.weekday())
    df1['throughput'] = df1['octets_passed'] * CONVERT_OCTETS
    return df1

def stream_sanitize(df):
    print "Heartbeat Threshold expected = ", HEARTBEAT_THRESH

    df['datetime'] = pd.to_datetime(df['end_time'])
    # ADJUST DATETIME
    df1['datetime']-=datetime.timedelta(hours=6)

    #df1 = df1[df1['end_time'] <= DATE_END]
    #df2 = df2[df2['end_time'] <= DATE_END]

    # get only devices with more than thresh entries in timeslots
    #test_count = df.groupby('Device_number')['datetime'].count()
    test_count = df.groupby('Device_number')['datetime'].unique().apply(lambda x: len(x))
    best_test = (test_count[ test_count > HEARTBEAT_THRESH ]).index

    # filter devices and save to pkl (only octets and end_time)
    df_test = df[ df['Device_number'].isin(best_test) ]


    return df_test

def save_by_direction(df, name='test', direction='dw'):
    FINALFOLDER="/data/users/sarthak/comcast-data/separated/final_"+direction+"/"
    if not os.path.exists(FINALFOLDER):
        os.makedirs(FINALFOLDER)


    if direction == 'dw':
        df_test = df[df['service_direction'] == 1]
    else:
        df_test = df[df['service_direction'] == 2]

    df_test[['name', 'Device_number','datetime', 'octets_passed', 'throughput', 'day', 'time']].to_pickle(FINALFOLDER + name + '.pkl')

    print "Done saving "+name+" "+direction+", Unique devices = ", len(df_test.Device_number.unique())

    return

def main2(name = 'test', direction='dw'):
    DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_"+direction+"/"
    FINALFOLDER="/data/users/sarthak/comcast-data/separated/final_"+direction+"/"
    if not os.path.exists(FINALFOLDER):
        os.makedirs(FINALFOLDER)

    if name=='test':
        df = pd.read_pickle(DATAFOLDER+"test.pkl")
    else:
        df = pd.read_pickle(DATAFOLDER+"control.pkl")

    df2 = stream_sanitize(df)
    df2[['name', 'Device_number','datetime', 'octets_passed', 'throughput', 'day', 'time']].to_pickle(FINALFOLDER + name + '.pkl')
    print "DONE"
    return


def main(name='test'):
    DATAFOLDER = "/data/users/sarthak/comcast-data/data/"
    header_row= ['Device_number', 'end_time', 'date_service_created',
                'service_class_name', 'cmts_inet', 'service_direction',
                'port_name', 'octets_passed', 'device_key',
                'service_identifier']

    if name == 'test':
        datafilename = '250-test.dat'
        df = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time', 'service_direction'], as_index=False).sum().reset_index()
        df['name'] = datafilename
        # SANITIZE
        print "Unique devices before sanitize "+datafilename+" = ", len(df['Device_number'].unique())
        df2 = stream_sanitize(df)
        print "Unique devices in "+datafilename+" = ", len(df['Device_number'].unique())

    else:
        all_df = defaultdict(int)
        for datafilename in ['control'+str(x)+'.dat' for x in range(1,9)]:
            df_temp = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time', 'service_direction'], as_index=False).sum()
            # SANITIZE can be applied here instead: only control4 contributes
            # (1500 devices)
            #df_temp = stream_sanitize(df_temp)
            all_df[ datafilename ] = df_temp
            print "Unique devices before sanitize "+datafilename+" = ", len(df_temp['Device_number'].unique())
        df = pd.concat(all_df).reset_index().rename(columns={'level_0':'name'})
        # SANITIZE
        df2 = stream_sanitize(df)
        print "Unique devices len = ", len(df['Device_number'].unique())

    df3 = add_extra_cols(df2)
    print "df features ", len(df3), df3.columns, name
    save_by_direction(df3, name, 'up')
    save_by_direction(df3, name, 'dw')

    return


def sanitize(df_t, df_c, direction='dw', THRESH=0.8):
    FINALFOLDER="/data/users/sarthak/comcast-data/separated/final_"+direction+"/"
    if not os.path.exists(FINALFOLDER):
        os.makedirs(FINALFOLDER)

    if direction == 'dw':
        df1 = df_t[df_t['service_direction']==1]
        df2 = df_c[df_c['service_direction']==1]
    elif direction == 'up':
        df1 = df_t[df_t['service_direction']==2]
        df2 = df_c[df_c['service_direction']==2]
    else:
        print "Error"

    # max avail threshold = 0.8 of full range
    ts = pd.date_range(DATE_START, DATE_END, freq='15min')
    HEARTBEAT_THRESH = THRESH * len(ts)
    print "Heartbeat Threshold expected = ", HEARTBEAT_THRESH
    #df1 = df1[df1['end_time'] <= DATE_END]
    #df2 = df2[df2['end_time'] <= DATE_END]

    # get only devices with more than thresh entries in timeslots
    test_count = df1.groupby('Device_number')['datetime'].count()
    cont_count = df2.groupby('Device_number')['datetime'].count()
    best_test = (test_count[ test_count > HEARTBEAT_THRESH ]).index
    best_cont = (cont_count[ cont_count > HEARTBEAT_THRESH ]).index

    # filter devices and save to pkl (only octets and end_time)
    df_test = add_extra_cols( df1[ df1['Device_number'].isin(best_test) ] )
    df_cont = add_extra_cols( df2[ df2['Device_number'].isin(best_cont) ] )

    df_test[['name', 'Device_number','datetime', 'octets_passed', 'throughput', 'day', 'time']].to_pickle(FINALFOLDER + 'test.pkl')
    df_cont[['name', 'Device_number','datetime', 'octets_passed', 'throughput', 'day', 'time']].to_pickle(FINALFOLDER + 'control.pkl')

    print "Sanitized test size = ", len(df_test['Device_number'].unique())
    print "Sanitized cont size = ", len(df_cont['Device_number'].unique())

    return

def choose_random(df, NUMBER=1000):
    devices = df.Device_number.unique()
    np.random.shuffle( devices )
    df_c = df[df.Device_number.isin(devices[:NUMBER])]
    return df_c

def main3(name='test'):

    for direction in ['dw', 'up']:
        FINALFOLDER="/data/users/sarthak/comcast-data/separated/final_"+direction+"/"
        FINALFOLDER1k="/data/users/sarthak/comcast-data/separated/final_1k_"+direction+"/"
        if not os.path.exists(FINALFOLDER1k):
            os.makedirs(FINALFOLDER1k)

        df = pd.read_pickle(FINALFOLDER + name + ".pkl")
        print "unique dev in full "+name+" "+direction+" set: ", len(df.Device_number.unique())
        df1 = choose_random(df)
        print "unique dev in chosen "+name+" "+direction+" set: ", len(df1.Device_number.unique())
        df1.to_pickle(FINALFOLDER1k + name + '.pkl')

    print "DONE"
    return

if __name__=='__main__':
    #sanitize('dw', 0.8)
    #sanitize('up', 0.8)
    #df1 = load_df('test', 0)
    #df2 = load_df('control', 0)
    #sanitize(df1, df2, 'dw', 0.8)
    #sanitize(df1, df2, 'up', 0.8)
    #main()
    main('test')
    main('control')
    main3('test')
    main3('control')
