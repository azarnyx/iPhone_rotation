import pandas as pd
import pylab as plt


df = pd.read_csv("../data/measurements.csv")

li = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)', 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)']

cls = [(117, 178, 251), (251, 70, 86), (83, 212, 104), (117, 178, 251), (251, 70, 86), (83, 212, 104)]
cls = [tuple([i/255 for i in lj]) for lj in cls]

fig, ax = plt.subplots(len(li), 1, figsize=(7.5, 15))

for ni, i in enumerate(li[0:len(ax)]):
    line = ax[ni].plot(df.loc[:, 'loggingTime(txt)'], df.loc[:, i], color=cls[ni], label=i, lw=1.5)
    if ni == 2:
        ax[ni].set_ylim([-1.1,1.1])
    l = ax[ni].legend(loc='upper right',fontsize=15)
    l.get_frame().set_alpha(None)
plt.tight_layout()
plt.savefig('../pics/Figure_0.png', dpi=500)    
plt.show()
