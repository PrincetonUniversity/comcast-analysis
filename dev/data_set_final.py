import pandas as pd
import os, sys
from collections import defaultdict

'''
Aim: select only those control devices that exist throughout
the date range 30 sept to 29 dec
'''

DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_dw"
#DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_up/"
FINALFOLDER='/data/users/sarthak/comcast-data/final_dw/'
#FINALFOLDER='/data/users/sarthak/comcast-data/final/up/'
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
    df_control[['Device_number','octets_passed','datetime']].to_pickle("../../final/control.pkl")

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

def load_test_and_control(direction, processed=1):
    if processed:
        DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_"+direction+"/"
        df1 = pd.read_pickle(DATAFOLDER+"test.pkl")
        df2 = pd.read_pickle(DATAFOLDER+"control.pkl")
        return df1, df2
    else:
        DATAFOLDER = "/data/users/sarthak/comcast-data/data/"
        header_row= ['Device_number', 'end_time', 'date_service_created',
                    'service_class_name', 'cmts_inet', 'service_direction',
                    'port_name', 'octets_passed', 'device_key',
                    'service_identifier']

        datafilename = '250-test.dat'
        df1 = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time'], as_index=False).sum().reset_index()
        df1['name'] = datafilename
        print "Done test set load"

        all_df = defaultdict(int)
        for datafilename in ['control'+str(x)+'.dat' for x in range(1,9)]:
            print "load " + datafilename
            all_df[datafilename] = pd.read_csv(DATAFOLDER + datafilename, delimiter='|', names=header_row)[['Device_number', 'end_time', 'service_direction', 'octets_passed']].groupby(['Device_number','end_time'], as_index=False).sum()
        df2 = pd.concat(all_df).reset_index().rename(columns={'level_0':'name'})
        print "Done control set load"

        if direction == 'dw':
            return df1[df1['service_direction']==1], df2[df2['service_direction']==1]
        elif direction == 'up':
            return df1[df1['service_direction']==2], df2[df2['service_direction']==2]

        return None, None


def sanitize(direction='dw', THRESH=0.8):
    FINALFOLDER="/data/users/sarthak/comcast-data/final_"+direction+"/"
    if not os.path.exists(FINALFOLDER):
        os.makedirs(FINALFOLDER)

    # load test and control sets
    df1, df2 = load_test_and_control(direction, 0)

    # max avail threshold = 0.8 of full range
    ts = pd.date_range('2014-09-30', '2014-12-29', freq='15min')
    HEARTBEAT_THRESH = THRESH * len(ts)

    # get only devices with more than thresh entries in timeslots
    test_count = df1.groupby('Device_number')['datetime'].count()
    cont_count = df2.groupby('Device_number')['datetime'].count()
    best_test = (test_count[ test_count > HEARTBEAT_THRESH ]).index
    best_cont = (cont_count[ cont_count > HEARTBEAT_THRESH ]).index

    # filter devices and save to pkl (only octets and end_time)
    df_test = df1[ df1['Device_number'].isin(best_test) ]
    df_cont = df2[ df2['Device_number'].isin(best_cont) ]
    df_test[['name', 'Device_number','end_time', 'octets_passed']].to_pickle(FINALFOLDER + 'test.pkl')
    df_cont[['name', 'Device_number','end_time', 'octets_passed']].to_pickle(FINALFOLDER + 'control.pkl')

    print "Sanitized test size = ", len(df_test['Device_number'].unique())
    print "Sanitized cont size = ", len(df_cont['Device_number'].unique())

    return

if __name__=='__main__':
    sanitize('dw', 0.8)
    sanitize('up', 0.8)
