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
