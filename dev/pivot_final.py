import pandas as pd
import os

DATA = "/data/users/sarthak/comcast-data/"

for folder in os.listdir(DATA + "final/"):
    for filename in os.listdir(DATA + "final/"+folder+"/"):
        filepath = DATA + "final/"+folder+"/"+filename
        print filepath
        # Device_number 547 was repeated exactly in control5 and control8 - drop it!
        df1 = pd.read_pickle(filepath)[['Device_number','datetime','octets_passed']].drop_duplicates()
        df2 = df1.pivot(index='datetime', columns='Device_number', values='octets_passed')
        outfolder = DATA + "final_processed/"+folder+"/"
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        df2.to_pickle(DATA + "final_processed/"+folder+"/"+filename)
        del df1, df2

def get_weekday_time_groups():

    '''run on dp7 in final folder'''
    CONVERT_OCT = 8.0 / (15*60 * 1024)
    for folder in ['final_dw/', 'final_up/']:
        for dataset in ['test', 'control']:
            df = pd.read_pickle(folder + dataset + '.pkl')
            mean_time_day = df.groupby(['Device_number', 'day', 'time'])['octets_passed'].mean()
            tp = (mean_time_day * CONVERT_OCT).unstack(1)
            weekdays = tp[[0,1,2,3,4]].mean(1).unstack(-2)
            weekends = tp[[5,6]].mean(1).unstack(-2)
            weekdays.to_pickle(folder + dataset + "_weekday.pkl")
            weekends.to_pickle(folder + dataset + "_weekend.pkl")
    return
