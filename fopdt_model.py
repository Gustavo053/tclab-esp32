######################################################
# FOPDT model                                        #
######################################################
import numpy as np
import matplotlib.pyplot as plt

Kp = 1      # degC/%
tauP = 130.0  # seconds
thetaP = 13   # seconds (integer)
Tss = 30      # degC (ambient temperature)
Qss = 0       # % heater
dt = 1 #1s
Tpl = []
Tpl.append(30)
i = 1
tMax = 8*60
nSamples = tMax/dt
# Simulate one time step with linear FOPDT model
z = np.exp(-dt/tauP)
Q1 = np.ones(int(nSamples)) * 0
Q1[30:] = 70.0
while (i <= nSamples):
    Tpl.append((Tpl[i-1]-Tss) * z + (Q1[max(0, i-int(thetaP)-1)]-Qss)*(1-z)*Kp + Tss)      
    i += 1

time = np.linspace(0, tMax, int(nSamples))
plt.plot(time, Tpl[0:480])
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('y (output)')
plt.show()