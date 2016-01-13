import pandas as pd

df1 = pd.read_pickle('/data/users/sarthak/comcast-data/filtered/250-test_dw.pkl')
test_dev = list(df1['Device_number'].unique())

print "Test dev", len(df1['Device_number'].unique())

control_dev = []
for num in range(8):
    num_str = str(num + 1)
    df2 = pd.read_pickle('/data/users/sarthak/comcast-data/filtered/control'+num_str+'_dw.pkl')
    print num_str, len(df2.Device_number.unique())
    control_dev.extend(list(df2.Device_number.unique()))

print "Control dev", len(set(control_dev))

total_dev = list(control_dev) + list(test_dev)
print "Total dev", len(set(total_dev))
