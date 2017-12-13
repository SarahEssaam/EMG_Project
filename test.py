from EMG_process import EMG_Process
import numpy as np
import matplotlib.pyplot as plt

#Read file into signal_init
EMG = EMG_Process("Data.txt",T=20,DiffTh=12.65**5)
timeStamps, templates = EMG.process(method="method1")
plt.figure(1)
EMG.plot(signal="init")
plt.figure(2)
EMG.plot_templates()
plt.show()
