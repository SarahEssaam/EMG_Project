from EMG_process import EMG_Process
import matplotlib.pyplot as plt

EMG = EMG_Process("Data.txt",T=20,DiffTh=12.65**5)
#Q1
timeStamps, templates = EMG.process(method="method1",debug=1)

plt.figure(1)
EMG.plot(signal="init",start=30000,end=35000)
plt.savefig('DetectedMUAP.png',bbox_inches='tight')

plt.figure(2)
EMG.plot_templates()
plt.savefig('Templates.png',bbox_inches='tight')

#Q2
timeStamps, templates = EMG.process(method="kmeans")

plt.figure(3)
EMG.plot(signal="init",start=30000,end=35000)
plt.savefig('DetectedMUAP_K.png',bbox_inches='tight')

plt.figure(4)
EMG.plot_templates()
plt.savefig('Templates_K.png',bbox_inches='tight')

