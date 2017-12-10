import EMG_Functions as EMG
import numpy as np
import matplotlib.pyplot as plt

#Read file into signal_init
file = open('Data.txt', 'r')
signal_strings = file.read().split('\n')
signal_init = np.asarray(signal_strings, dtype=np.float)
file.close()
#Smooth signal
signal_smooth = EMG.smooth(signal_init,20)
#visualize the signal and choose 500 to 2500 as an error region to get std dev for
#plt.plot(signal_init[:3000])
threshold = EMG.getThreshold(signal_smooth[500:2500])
#get timeStamps and each MUAP as a row in MUAP array
timeStamps, MUAPs = EMG.breakSignal(signal_smooth,signal_init,threshold)
templates, labels = EMG.decompose(MUAPs, 12.65**5)
EMG.plot(signal_init,timeStamps,labels,start=0000,end=1000)
EMG.plot_templates(templates)
plt.show()
