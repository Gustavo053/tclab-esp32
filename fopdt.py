import numpy as np
import matplotlib.pyplot as plt
import time
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

n = 480  # Number of second time points (8 min)
tm = np.linspace(0,n,n+1) # Time values

# data
board.send_sysex(0x02, [32])
time.sleep(0.1) #Esse sleep é necessário para dar tempo do valor da variável temperatura ser atualizado
T1 = [temperature]
QPlot = np.zeros(n+1) #for plotting the percentage value
QPlot[30:] = 70.0 #for plotting the percentage value
Q1 = np.zeros(n+1)
Q1[30:] = 178.5 #70.0%
for i in range(n):
    datasToWrite = []
    datasToWrite.append(12)
    datasToWrite.append(0)
    datasToWrite.append(49)
    datasToWrite.append(8)

    datasToWrite.append(int(Q1[i]))
    board.send_sysex(0x04, datasToWrite)
    time.sleep(1)

    board.send_sysex(0x02, [32])
    print(temperature)
    T1.append(temperature)

# Save data file
data = np.vstack((tm,Q1,T1)).T
np.savetxt('Step_Response.csv',data,delimiter=',',\
            header='Time,Q1,T1',comments='')

# Create Figure
plt.figure(figsize=(12,8))
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(tm/60.0,T1,'r.',label=r'$T_1$')
plt.ylabel(r'Temp ($^oC$)')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(tm/60.0,QPlot,'b-',label=r'$Q_1$')
plt.ylabel(r'Heater (%)')
plt.xlabel('Time (min)')
plt.legend()
plt.savefig('Step_Response.png')
plt.show()
