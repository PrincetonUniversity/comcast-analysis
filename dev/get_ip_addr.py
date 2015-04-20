import pandas as pd
import multiprocess as mp

TESTSET = '250-test.dat'
CONTROLSETS = ["control"+str(x)+".dat" for x in range(1,9)]
SETS = CONTROLSETS[:].append(TESTSET)
SANITIZEDPATH = '/data/users/sarthak/comcast-data/separated/'

def get_device_ip_dict(setname):
    df = pd.read_pickle(SANITIZEDPATH + setname )

    df.groupby()
    return

if __name__ == '__main__':
    jobs = []
    for dataset in SETS:
        setname = dataset.split('.')[0]
        if 'test' in setname:
            setname += '/test.pkl'
        else:
            setname += '/control.pkl'

        p = mp.Process( target=get_device_ip_dict, args=(setname,) )
        jobs.append(p)
        p.start()
