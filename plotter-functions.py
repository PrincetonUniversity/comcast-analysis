    def plotBytesHist(self):
    # Histogram of all octets_passed in the dataset
        fig3, ax3 = plt.subplots(2,2, figsize=(16,16), sharex=1, sharey=1)
        self.test_up['octets_passed'].hist(ax=ax3[0][0])
        self.test_dw['octets_passed'].hist(ax=ax3[0][1])
        self.control_up['octets_passed'].hist(ax=ax3[1][0], color='g')
        self.control_dw['octets_passed'].hist(ax=ax3[1][1], color='g')

        ax3[0][0].set_title('250-test_up')
        ax3[0][1].set_title('250-test_dw')
        ax3[1][1].set_title('control1_dw')
        ax3[1][0].set_title('control1_up')
        fig3.tight_layout()
        fig3.savefig(OUTPUTPATH+'all-bytes-hist')
        return

    def plotBytesCDF(self):
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10), sharex=1)

        x, y = getSortedCDF(list(self.test_up['octets_passed']))
        ax1[0].plot(x,y, color='b', label='250-test_up')
        x, y = getSortedCDF(list(self.control_up['octets_passed']))
        ax1[0].plot(x,y, color='g', label='control1_up')

        x, y = getSortedCDF(list(self.test_dw['octets_passed']))
        ax1[1].plot(x,y, color='b', label='250-test_dw')
        x, y = getSortedCDF(list(self.control_dw['octets_passed']))
        ax1[1].plot(x,y, color='g', label='control1_dw')

        ax1[1].set_xlabel("Bytes per 15-min time slot")
        ax1[0].set_ylabel("CUMULATIVE COUNT UPLINK")
        ax1[1].set_ylabel("CUMULATIVE COUNT DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH+'all-bytes-cdf')
        return

    def plotNonZeroBytes(self):
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))
        df = self.test_up[self.test_up['octets_passed'] > 0]['octets_passed']
        x, y = getSortedCDF(list(df))
        ax1[0].plot(x,y, color='b', label='250-test_up')
        df = self.control_up[self.control_up['octets_passed'] > 0]['octets_passed']
        x, y = getSortedCDF(list(df))
        ax1[0].plot(x,y, color='g', label='control1_up')

        df = self.test_dw[self.test_dw['octets_passed'] > 0]['octets_passed']
        x, y = getSortedCDF(list(df))
        ax1[1].plot(x,y, color='b', label='250-test_dw')
        df = self.control_dw[self.control_dw['octets_passed'] > 0]['octets_passed']
        x, y = getSortedCDF(list(df))
        ax1[1].plot(x,y, color='g', label='control1_dw')

        ax1[1].set_xlabel("Non Zero Bytes per 15-min time slot")
        ax1[0].set_ylabel("CDF UPLINK")
        ax1[1].set_ylabel("CDF DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH+'nonzero-bytes-cdf')

        return

    def plotPeakBytesPerDevice(self):
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))

        labels = iter(['250-test_up','control1_up','250-test_dw',
                       'control1_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])

        for df_temp in [self.test_up, self.control_up, self.test_dw,
                        self.control_dw]:
            df = df_temp.sort('octets_passed', ascending=False).groupby(
                ['Device_number'], as_index=False).first()['octets_passed']
            # Also display the mean in the legend?
            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % df.mean()
                lab += ': '+ calcMean
            x, y = getSortedCDF(list(df))
            ax1[next(axis_ctr)].plot(x,y, color=next(colors),
                                     label=lab+": "+calcMean)

        ax1[1].set_xlabel("Peak Bytes per Device")
        ax1[0].set_ylabel("CDF UPLINK")
        ax1[1].set_ylabel("CDF DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH+'peakBytesPerDevice-cdf')
        return

    def plotPeakBytesPerDayPerDevice(self):
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))

        labels = iter(['250-test_up','control1_up','250-test_dw',
                       'control1_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])

        for df_temp in [self.test_up, self.control_up, self.test_dw,
                        self.control_dw]:
            df = df_temp.sort('octets_passed', ascending=False).groupby(
                ['Device_number', 'date'], as_index=False).first()['octets_passed']
            x, y = getSortedCDF(list(df))
            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % df.mean()
                lab += ': '+ calcMean
            ax1[next(axis_ctr)].plot(x,y, color=next(colors), label=lab)

        ax1[1].set_xlabel("Peak Bytes per Day per Device")
        ax1[0].set_ylabel("CUMULATIVE COUNT UPLINK")
        ax1[1].set_ylabel("CUMULATIVE COUNT DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH+'peakBytesPerDayPerDevice-cdf')
        return

    def plotPeakBytesPerDevicePerTimeSlot(self):
        fig1, ax1 = plt.subplots(2,1, figsize=(8,10))

        labels = iter(['250-test_up','control1_up','250-test_dw',
                       'control1_dw'])
        colors = iter(['b', 'g', 'b', 'g'])
        axis_ctr = iter([0,0,1,1])

        for df_temp in [self.test_up, self.control_up, self.test_dw,
                        self.control_dw]:
            df = df_temp.sort('octets_passed', ascending=False).groupby(
                ['Device_number', 'time'], as_index=False).first()
            ['octets_passed']
            x, y = getSortedCDF(list(df))
            lab = next(labels)
            if self.DISPLAY_MEAN:
                calcMean = '%.2E' % df.mean()
                lab += ': '+ calcMean
            ax1[next(axis_ctr)].plot(x,y, color=next(colors), label=lab)

        ax1[1].set_xlabel("Peak Bytes per Device per Time-Slot")
        ax1[0].set_ylabel("CUMULATIVE COUNT UPLINK")
        ax1[1].set_ylabel("CUMULATIVE COUNT DOWNLINK")
        ax1[0].grid(1)
        ax1[1].grid(1)
        ax1[0].legend(loc='best')
        ax1[1].legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(OUTPUTPATH+'peakBytesPerDevicePerTimeSlot-cdf')
        return


