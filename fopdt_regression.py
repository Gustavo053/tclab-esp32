import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyUSB0')
it = util.Iterator(board)
it.start()

print('✔ board ready!')

temperature = 0

def handle_read_write(*data):
    rawV = 0

    for i in data:
        rawV = rawV + i

    global temperature
    temperature = (rawV * (5000 / 1023)) / 10


board.add_cmd_handler(0x02, handle_read_write)

try:
    # read Step_Response.csv if it exists
    data = pd.read_csv('Step_Response.csv')
    tm = data['Time'].values
    Q1 = data['Q1'].values
    T1 = data['T1'].values
except:
    # generate data only once
    n = 840  # Number of second time points (14 min)
    tm = np.linspace(0,n,n+1) # Time values
    board.send_sysex(0x02, [32])
    T1 = [temperature]

    #for plotting the percentage value
    Qplot = np.zeros(n+1)
    Qplot[30:] = 35.0 
    Qplot[270:] = 70.0
    Qplot[450:] = 10.0
    Qplot[630:] = 60.0
    Qplot[800:] = 0.0

    Q1 = np.zeros(n+1)
    Q1[30:] = 89.25 #35% heater
    Q1[270:] = 178.5 #70% heater
    Q1[450:] = 25.5 #10% heater
    Q1[630:] = 153.0 #60% heater
    Q1[800:] = 0.0 #0% heater
    for i in range(n):
        datasToWrite = []
        datasToWrite.append(12)
        datasToWrite.append(0)
        datasToWrite.append(49)
        datasToWrite.append(8)

        print('Q1[i]: ', Q1[i])
        datasToWrite.append(int(Q1[i]))
        board.send_sysex(0x04, datasToWrite)
        time.sleep(1)

        board.send_sysex(0x02, [32])
        print(temperature)
        T1.append(temperature)
    # Save data file
    data = np.vstack((tm,Q1,T1)).T
    np.savetxt('Step_Response_Regression.csv',data,delimiter=',',\
               header='Time,Q1,T1',comments='')

# Create Figure
plt.figure(figsize=(10,7))
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(tm/60.0,T1,'r.',label=r'$T_1$')
plt.ylabel(r'Temp ($^oC$)')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(tm/60.0,Qplot,'b-',label=r'$Q_1$')
plt.ylabel(r'Heater (%)')
plt.xlabel('Time (min)')
plt.legend()
plt.savefig('Step_Response_Regression.png')
plt.show()