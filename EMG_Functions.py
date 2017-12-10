import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn.cluster import KMeans


def smooth(signal, N):
    signal = np.absolute(signal)
    kernel = np.ones(N, dtype=float) * (1 / N)
    signal_smoothed = np.zeros_like(signal)
    signal = np.append(signal, np.zeros(N - 1))
    for i in range(signal_smoothed.size):
        signal_smoothed[i] = np.sum(signal[i:i + N] * kernel)
    return signal_smoothed

def getThreshold(signal_smooth):
    return 3*np.std(signal_smooth)

def split_above_threshold(smooth_signal,signal, threshold):
    mask = np.concatenate(([False], smooth_signal > threshold, [False] ))
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
    return [signal[idx[i]:idx[i+1]] for i in range(0,len(idx),2)]#if idx[i+1]-idx[i] > 20 ]

def breakSignal(smooth_signal,signal,threshold):
    array_above_indices = np.where(smooth_signal > threshold)
    above_signals = split_above_threshold(smooth_signal,signal,threshold)
    i = 0
    time_stamps = np.array([],dtype=np.int)
#Shoould i be working on the original signal?
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
    MUAPs = [ signal[i-10:i+11] for i in time_stamps]
    return time_stamps, np.array(MUAPs)

def decompose(MUAPs, Dthresh):
    # each MUAP signal is represented as a row in the MUAPs array
    N = MUAPs.shape[0]
    labels = np.zeros(N,dtype=int)
    templates = np.array([],dtype=np.float)
    for i in range(N):
        MUAP = (MUAPs[i]).reshape(1,MUAPs.shape[1])
        if i ==0 :
            templates = MUAP
            labels[0] = 0
            continue
        dist = np.sum(np.square(MUAP - templates),axis = 1)
        #see if more than one distance is less than Dthreshold
        count = np.count_nonzero(dist < Dthresh)
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

def plot(signal, time_stamps, labels, start=0,end=0):
    #use 7 colors for first 6 MUs and black for the rest
    color = {0:'r', 1:'b', 2:'g', 3:'c', 4:'m', 5:'y',6:'k'}
    if end==0:
        end = signal.shape[0]
    plt.plot(range(start, end), signal[start:end], linestyle='-', color='k')
    time_stamps = time_stamps[time_stamps >= start]
    time_stamps = time_stamps[time_stamps < end]
    max = np.max(signal[start:end])
    for i in range(time_stamps.shape[0]):
        plt.plot(time_stamps[i], signal[time_stamps[i]], marker='*', linestyle='None', color=color.get(labels[i],'k'))

def plot_templates(templates):
    color = {0: 'r', 1: 'b', 2: 'g', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}
    T = len(templates)
    for i in (range(T)):
        plt.subplot(ceil(T/3),3,i+1)
        plt.plot(templates[i],color=color.get(i,'k'))
#The output templates are not centered at 10! cuz the smoothed ones are the ones centered at 10