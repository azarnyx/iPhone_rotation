import pandas as pd
import pylab as plt
import numpy as np
from tqdm import tqdm

df = pd.read_csv("../data/measurements.csv")

# 0. Initialize variablles
e0 = np.array((1, 0, 0), np.longdouble)
e1 = np.array((0, 1, 0), np.longdouble)
e2 = np.array((0, 0, 1), np.longdouble)
xx = np.array((1, 0, 0), np.longdouble)
yy = np.array((0, 1, 0), np.longdouble)
zz = np.array((0, 0, 1), np.longdouble)
dt = 0.01
time = 0
li_out = []
Time = df.loc[:, 'loggingTime(txt)']
ran = np.arange(0, Time.iloc[-1], dt)
i=0

for time in tqdm(ran):
    # 1. Interpolate projections of rotation in time
    o0 = np.interp(time, Time, df['gyroRotationX(rad/s)'])
    o1 = np.interp(time, Time, df['gyroRotationY(rad/s)'])
    o2 = np.interp(time, Time, df['gyroRotationZ(rad/s)'])

    # 2. Compute total rotational speed in iPhone coordinates (eq. 1)
    omega = o0 * e0 + o1 * e1 + o2 * e2

    # 3. update iPhone coordinates based on the rotation (eq. 2)
    e0 += dt * np.cross(omega, e0)
    e1 += dt * np.cross(omega, e1)
    e2 += dt * np.cross(omega, e2)
    
    # 4. Gram-Schmidt transformation on new iPhone coordinates
    e0 /= np.linalg.norm(e0)
    e1 -= np.dot(e1, e0) * e0
    e1 /= np.linalg.norm(e1)
    e2 -= np.dot(e2, e0) * e0
    e2 -= np.dot(e2, e1) * e1
    e2 /= np.linalg.norm(e2)

    if i % 10 == 0:
        # 5. Calculate alpha (eq. 3)

        e0_proj = e0 - np.dot(e0, zz) * zz
        alphax = np.arccos(np.dot(xx, e0_proj)/np.linalg.norm(e0_proj))
        alphay = np.arccos(np.dot(yy, e0_proj)/np.linalg.norm(e0_proj))
        if alphay > np.pi/2:
            alpha = -alphax
        else:
            alpha = alphax
        
        # 6. output and visualization
        li_out.append([time, alpha])
    time += dt
    i += 1
    
df_out = pd.DataFrame(li_out, columns=['time', 'alpha'])
fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
ax.plot(df_out.time, df_out.alpha*180/np.pi, lw=1.5, label=r'$\alpha$')
ax.axhline(y=0, ls='--', lw=0.5, color='grey')
ax.set_xlabel('time, sec')
ax.set_ylabel(r'computed angle, °')
l = ax.legend(loc='upper left',fontsize=15)
l.get_frame().set_alpha(None)
plt.tight_layout()
plt.savefig('../pics/Figure_1.png', dpi=500)
plt.show()
