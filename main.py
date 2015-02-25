from dataplot import *

def getSortedCDF(data):
    """
    return x,y for a cdf given one dimensional unsorted data x
    """
    sorted_data = np.sort( data )
    YSCALE = len(sorted_data)
    yvals = [y/YSCALE for y in range( len(sorted_data) )]
    return sorted_data, yvals

def plotCDF(test, control):
    """
    Index([u'Device_number', u'end_time', u'octets_passed'], dtype='object')
    """
    fig1, ax1 = plt.subplots(1,1, figsize=(8,8))

    # test data
    x, y = getSortedCDF()



    return

def plotALLTimeSeries(df1, df2):
    """
    Index([u'Device_number', u'end_time', u'octets_passed'], dtype='object')
    """
    ctr = 0
    names = {0:'test', 1:'control'}
    for df in [df1, df2]:
        x = list(pd.to_date_time(df['end_time']))
        if method=='all':
            y = list(df['octets_passed'])
        if method=='max':

        calcMean = '%.2E' % np.mean(y)
        lab = names[ctr] + ': ' + calcMean
        ax1.plot_date(x, y, m=markers[ctr], c=colors[ctr],
                          alpha=0.3, label=lab)
    ax1.set_xlabel("DateTime")
    ax1.set_ylabel("All Bytes")
    ax1.grid(1)
    ax1.legend(loc='best')
    fig.tight_layout()
    fig1.savefig(OUTPUTPATH+figname)
    return

if __name__=='__main__':
    test = ComcastDataSet(FILENAME='250-test.dat', **CONST)
    control = ComcastDataSet(FILENAME='control1.dat', **CONST)

    #test.name = '250-test'
    #control.name = 'control1'

    test.load()
    control.load()

    # filter 1000 devices
    test.filter_by_device()
    control.filter_by_device()

    # get control set dates and filter test set accordingly
    dstart, dstop = control.get_date_range()
    print dstart, dstop
    test.filter_by_date(dstart, dstop)
    print test.up['end_time'].iloc[0], test.up['end_time'].iloc[-1]

    # SAVEPOINT
    # sum over all weird service_class_name
    print "SAVEPOINT"
    test.aggregate()
    test.save_processed()
    control.aggregate()
    control.save_processed()

    print "TEST", test.up.describe()
    print "CONTROL", control.up.describe()

# 0. Long Timeseries [Max|90ile|median] [Bytes groupby datetime]
    plotTimeSeries(test.up, control.up)


# 1. CDF of ALL Bytes

# 2. CDF [max|90%] [Bytes groupby Device]
# 3. Difference in mean[Bytes ALL] vs THRESHOLD
# 4. Long TimeSeries BoxPlot (over Devices)
