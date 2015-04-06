from __future__ import division
import pandas as pd
from collections import defaultdict

header_row = ["Device_number", "end_time", "date_service_created",
              "service_class_name", "cmts_inet", "service_direction",
              "port_name", "octets_passed", "device_key", "service_identifier"]

dfname = "250-test.dat"
df = pd.read_csv("/data/users/sarthak/comcast-data/data/"+dfname,
                 delimiter="|", names=header_row)

# UNIQUE DEVICE
print "Unique Devices, how many?"
print df['Device_number'].unique(), len(df['Device_number'].unique())

# For AVAILABILITY by Device and by Time Slot in raw set
#TODO
df2 = df.groupby(['Device_number', 'end_time'], as_index=False)[
    'octets_passed'].sum()
print "Data set length?"
print len(df2)

serv = df.groupby(['service_direction', 'service_class_name'], as_index=False)['octets_passed'].sum()
serv_up = serv[serv.service_direction==2][['service_class_name', 'octets_passed']]
serv_dw = serv[serv.service_direction==1][['service_class_name', 'octets_passed']]

print "Which service class name is prominent?"
print serv_up/serv_up.sum()
print serv_dw/serv_dw.sum()
