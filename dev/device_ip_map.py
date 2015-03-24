"""
Extract IPaddresses in the correct format out of comcast's *.dat file
for Roya to plot on a map
Current version uses the complete file
Later we may want to replace this with the 3000-4000 selected IPs only based
on availability

"""

import struct, socket, csv, os, glob
from collections import defaultdict
import pickle as pkl

# store all device, date, ip. Later use pandas to unique and calc availability
device_date_ip = defaultdict(list)
# dict to store int:ip mappings. Roya gets csv values from this.values()
iplong_to_ipaddr = defaultdict(int)

# running in comcast-analysis go to data/
DATAPATH = "../data/"
# following paths are relative to DATAPATH
IPADDRPATH = "ipaddr/"
DEVICE_DATE_IP_PATH = "device_date_ip/"


os.chdir(DATAPATH)
for filename in glob.glob("*.dat"):
#for filename in ["test-data-set.dat"]:
    print filename
    device_date_ip = defaultdict(list)
    iplong_to_ipaddr = defaultdict(int)
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        for row in csvreader:
            # add all entries unconditionally. Later create csv of unique
            # values only
            device_date_ip['Device_number'].append(row[0])
            device_date_ip['date'].append(row[1])

            if iplong_to_ipaddr.get(int(row[4]), 0) == 0:
                ipaddr = socket.inet_ntoa(struct.pack('!L', int(row[4])))
                iplong_to_ipaddr[int(row[4])] = ipaddr

            device_date_ip['ipaddr'].append( ipaddr )

    pkl.dump(device_date_ip, open(DEVICE_DATE_IP_PATH+filename.split('.')[0]+".pkl", 'w'))
    #pd.DataFrame(device_date_ip).to_pickle("device_date_ip/"+filename.split('.')[0]+".pkl")
    #pd.Series(iplong_to_ipaddr).to_csv("ipaddr/"+filename.split('.')[0]+".csv")
    with open(IPADDRPATH+filename.split('.')[0]+".csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(iplong_to_ipaddr.values())
