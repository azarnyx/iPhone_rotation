import pandas as pd
import pylab as plt


df = pd.read_csv("../data/measurements_new.csv")
df = df.loc[(df.loc[:, 'loggingTime(txt)'] < 113.) & (df.loc[:, 'loggingTime(txt)']>5.)]
li = ['gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)','accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']

labels = [r'$o_0$',r'$o_1$',r'$o_2$',r'$a_0$',r'$a_1$',r'$a_2$']

cls = [(117, 178, 251), (251, 70, 86), (83, 212, 104), (117, 178, 251), (251, 70, 86), (83, 212, 104)]
cls = [tuple([i/255 for i in lj]) for lj in cls]

fig, ax = plt.subplots(len(li), 1, figsize=(7.5, 15))

for ni, i in enumerate(li[0:len(ax)]):
    line = ax[ni].plot(df.loc[:, 'loggingTime(txt)'], df.loc[:, i], color=cls[ni], label=labels[ni], lw=1.5)
    l = ax[ni].legend(loc='upper left',fontsize=15)
    l.get_frame().set_alpha(None)
plt.tight_layout()
plt.savefig('../pics/data_pic.png', dpi=500)
plt.show()
