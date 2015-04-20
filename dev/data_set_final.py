import pandas as pd
import os

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

def sanitize(direction='dw'):
    DATAFOLDER="/data/users/sarthak/comcast-data/separated/full_"+direction+"/"
    FINALFOLDER="/data/users/sarthak/comcast-data/final_"+direction+"/"
    if not os.path.exists(FINALFOLDER):
        os.makedirs(FINALFOLDER)

    # load test and control sets
    df1 = pd.read_pickle(DATAFOLDER+"test.pkl")
    df2 = pd.read_pickle(DATAFOLDER+"control.pkl")

    # max avail threshold = 0.8 of full range
    ts = pd.date_range('2014-09-30', '2014-12-29', freq='15min')
    HEARTBEAT_THRESH = 0.8 * len(ts)

    # get only devices with more than thresh entries in timeslots
    test_count = df1.groupby('Device_number')['datetime'].count()
    cont_count = df2.groupby('Device_number')['datetime'].count()
    best_test = (test_count[ test_count > HEARTBEAT_THRESH ]).index
    best_cont = (cont_count[ cont_count > HEARTBEAT_THRESH ]).index

    # filter devices and save to pkl (only octets and end_time)
    df_test = df1[ df1['Device_number'].isin(best_test) ]
    df_cont = df2[ df2['Device_number'].isin(best_cont) ]
    df_test[['Device_number','end_time']].to_pickle(FINALFOLDER + 'test.pkl')
    df_cont[['Device_number','end_time']].to_pickle(FINALFOLDER + 'control.pkl')

    print "Sanitized test size = ", len(df_test['Device_number'].unique())
    print "Sanitized cont size = ", len(df_cont['Device_number'].unique())

    return

    #


