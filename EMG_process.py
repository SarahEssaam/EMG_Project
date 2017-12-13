import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn.cluster import KMeans
from sympy.integrals.heurisch import DiffCache


class EMG_Process:

    def __init__(self,dataFile,T,DiffTh):
        file = open(dataFile, 'r')
        signal_strings = file.read().split('\n')
        self.signal_init = np.asarray(signal_strings, dtype=np.float)
        file.close()
        self.T = T
        self.DiffTh = DiffTh
        self.color = {0: 'r', 1: 'b', 2: 'g', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}

    def process(self,method="method1"):
        # visualize the signal and choose 500 to 2500 as an error region to get std dev for
        # plt.plot(signal_init[:3000])
        threshold = self.__getThreshold(self.signal_init[500:2500])
        plt.figure(0)
        plt.plot(range(2000,5000),self.signal_init[2000:5000])
        plt.plot(range(2000,5000),np.ones(3000)*threshold)
        #print(threshold)
        print(self.signal_init[1293])
        self.__smooth()
        #threshold = self.__getThreshold(self.signal_smoothed[500:2500])
        #plt.plot(range(0,2000),np.ones(2000)*threshold)
        list_above_signals = self.__split_above_threshold(threshold)
        self.time_stamps, MUAPs = self.__breakSignal(list_above_signals,threshold)
        if method=="method1":
            #keep as self cuz we'll need themm in plotting
            self.templates, self.labels = self.__process_method1(MUAPs)
        else :
            self.templates, self.labels = self.__process_kmeans(self.time_stamps, MUAPs)
        print (self.time_stamps)
        return self.time_stamps, self.templates

    def __getThreshold(self,signal):
        return 3*np.std(signal)

    def __smooth(self):
        signal = np.absolute(self.signal_init)
        kernel = np.ones(self.T, dtype=float) * (1 / self.T)
        self.signal_smoothed = np.zeros_like(signal)
        signal = np.append(signal, np.zeros(self.T - 1))
        for i in range(self.signal_smoothed.size):
            self.signal_smoothed[i] = np.sum(signal[i:i + self.T] * kernel)
        print(self.signal_smoothed[1290:1295])
        plt.plot(range(2000,5000),self.signal_smoothed[2000:5000])

    def __split_above_threshold(self, threshold):
        mask = np.concatenate(([False], self.signal_smoothed > threshold, [False] ))
        idx = np.flatnonzero(mask[1:] != mask[:-1])
    #Remove false samples?
        '''
        delete_idx = np.array([])
        for i in range(0,idx.shape[0]-3,3):
            if (idx[i+1]-idx[i])<20:
                if (idx[i+2]-idx[i+1])<20:
                    if (idx[i+3]-idx[i+2])<20:
                        delete_idx = np.append(delete_idx,i+1,i+2)
        np.delete(idx,delete_idx)
        '''
        return [self.signal_init[idx[i]:idx[i+1]] for i in range(0,len(idx),2)]#if idx[i+1]-idx[i] > 20 ]

    def __breakSignal(self,above_signals,threshold):
        array_above_indices = np.where(self.signal_smoothed>threshold)
        i = 0
        time_stamps = np.array([],dtype=np.int)
    #Should i be working on the original signal?
        for array in above_signals:
            time_stamps = np.append(time_stamps, array_above_indices[0][np.argmax(array)+i])
            i += np.size(array)
        #Should sth be done if time_stamps[i+1] - time_stamps[i] <20 ?
        #Should we remove signals that last less than 20 samples above ?
    #Should MUAPs be extracted from smoothed or what?
        #now get the MUAPs centered around max
        #plt.plot(signal[0:430])
        #plt.plot(smooth_signal[0:430])
        #plt.plot(time_stamps[3],800,marker='*', linestyle='None', color='r')
        MUAPs = [ self.signal_init[i-10:i+11] for i in time_stamps]
        MUAPs = np.array(MUAPs)
        print(MUAPs.shape)
        return time_stamps, np.array(MUAPs)

    def __process_method1(self,MUAPs):
        # each MUAP signal is represented as a row in the MUAPs array
        N = MUAPs.shape[0]
        labels = np.zeros(N,dtype=int)
        templates = np.array([],dtype=np.float)
        for i in range(N):
            MUAP = MUAPs[i].reshape(1,self.T+1)
            if i ==0 :
                templates = MUAP.reshape(1,self.T+1)
                labels[0] = 0
                continue
            dist = np.sum(np.square(MUAP - templates),axis = 1)
            #see if more than one distance is less than Dthreshold
            count = np.count_nonzero(dist < self.DiffTh)
            if count == 0:
                #add MUAP as new template
                templates = np.vstack([templates, MUAP])
                labels[i] = templates.shape[0] - 1
            else:
                min = np.argmin(dist)
                labels[i] = min
                #update template with average
                templates[min] = (templates[min] + MUAP)/2
        return templates, labels

    def plot(self, signal="init", start=0,end=0):
        if signal == "init":
            signal = self.signal_init
        elif signal == "smooth":
            signal = self.signal_smoothed
        if end==0:
            end = signal.shape[0]
        plt.plot(range(start, end), signal[start:end], linestyle='-', color='k')
        time_stamps = self.time_stamps[self.time_stamps >= start]
        time_stamps = time_stamps[time_stamps < end]
        max = np.max(signal[start:end])
        for i in range(time_stamps.shape[0]):
            # use 7 colors for first 6 MUs and black for the rest
            plt.plot(time_stamps[i], signal[time_stamps[i]], marker='*', linestyle='None', color=self.color.get(self.labels[i],'k'))

    def plot_templates(self):
        T = len(self.templates)
        for i in (range(T)):
            plt.subplot(ceil(T/3),3,i+1)
            plt.plot(self.templates[i],color=self.color.get(i,'k'))
