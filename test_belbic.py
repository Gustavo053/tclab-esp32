#Pinout range of ESP32 board mapped by firmata-esp32

#define IS_PIN_DIGITAL(p)       (((p) >= 2 && (p) <= 5) || ((p) >= 13 && (p) <= 27))
#define IS_PIN_ANALOG(p)        (((p) >= 32 && (p) <= 32 + TOTAL_ANALOG_PINS))
#define IS_PIN_PWM(p)           digitalPinHasPWM(p) // all gpios in digital

#OBS: Para este teste, o produto da resolução PWM no arquivo FirmataExt.cpp foi alterado de 1000 para 10.
#     Esse alteração foi feita para que a frequência no pino do ESP32 seja a mesma do arduino uno: 490Hz.

from pyfirmata import Arduino, util
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from BELBIC import belbic

board = Arduino('/dev/ttyUSB0')
it = util.Iterator(board)
it.start()

print('✔ board ready!')

temperature = 0

def handle_temperature(*data):

    rawV = 0

    for i in data:
        rawV = rawV + i


    global temperature
    temperature = (rawV * (5000 / 1023)) / 10

    # print(temperature)


board.add_cmd_handler(0x02, handle_temperature)

def pid(sp, pv, pv_last, ierr, dt):
    Kc = 10.0  # K/%Heater
    tauI = 50.0  # sec
    tauD = 1.0  # sec
    # Parameters in terms of PID coefficients
    KP = Kc
    KI = Kc/tauI
    KD = Kc*tauD
    # ubias for controller (initial heater)
    op0 = 0
    # upper and lower bounds on heater level
    ophi = 100
    oplo = 0
    # calculate the error
    error = sp-pv
    # calculate the integral error
    ierr = ierr + KI * error * dt
    # calculate the measurement derivative
    dpv = (pv - pv_last) / dt
    # calculate the PID output
    P = KP * error
    I = ierr
    D = -KD * dpv
    op = op0 + P + I + D
    # implement anti-reset windup
    if op < oplo or op > ophi:
        I = I - KI * error * dt
        # clip output
        op = max(oplo, min(ophi, op))
    # return the controller output and PID terms
    return [op, P, I, D]

# save txt file with data and set point
# t = time
# u1,u2 = heaters
# y1,y2 = tempeatures
# sp1,sp2 = setpoints


def save_txt(t, u1, u2, y1, y2, sp1, sp2):
    data = np.vstack((t, u1, u2, y1, y2, sp1, sp2))  # vertical stack
    data = data.T  # transpose data
    top = ('Time (sec), Heater 1 (%), Heater 2 (%), '
           'Temperature 1 (degC), Temperature 2 (degC), '
           'Set Point 1 (degC), Set Point 2 (degC)')
    np.savetxt('data.txt', data, delimiter=',', header=top, comments='')

def save_data_belbic(y, u, erro):
    data = np.vstack((y, u, erro))
    data = data.T
    top = ('Temperature (degC), Heater (%), error')
    np.savetxt('data_belbic.csv', data, delimiter=',', header=top, comments='')


######################################################
# FOPDT model                                        #
######################################################

#Get in pidtuner.com
Kp = 0.34953822182155675
tauP = 113.55168586918873
thetaP = 31.945949604357498
Tss = 30
Qss = 0

#Get in regression experiment
# Kp = 0.6747566878044584
# tauP = 172.3375915827159
# thetaP = 29.130396322742243
# Tss = 30
# Qss = 0

#Original
# Kp = 0.5      # degC/%
# tauP = 120.0  # seconds
# thetaP = 10   # seconds (integer)
# Tss = 23      # degC (ambient temperature)
# Qss = 0       # % heater

######################################################
# Energy balance model                               #
######################################################


def heat(x, t, Q):
    # Parameters
    Ta = 23 + 273.15   # K
    U = 10.0           # W/m^2-K
    m = 4.0/1000.0     # kg
    Cp = 0.5 * 1000.0  # J/kg-K
    A = 12.0 / 100.0**2  # Area in m^2
    alpha = 0.01       # W / % heater
    eps = 0.9          # Emissivity
    sigma = 5.67e-8    # Stefan-Boltzman

    # Temperature State
    T = x[0]

    # Nonlinear Energy Balance
    dTdt = (1.0/(m*Cp))*(U*A*(Ta-T)
                         + eps * sigma * A * (Ta**4 - T**4)
                         + alpha*Q)
    return dTdt

######################################################
# Do not adjust anything below this point            #
######################################################

# Turn LED on
# print('LED On')
# a.LED(100)

# Run time in minutes
run_time = 15

# Number of cycles
loops = int(60.0*run_time)
tm = np.zeros(loops)

# Temperature
# set point (degC)
Tsp1 = np.ones(loops) * 35
Tsp1[60:] = 50.0
Tsp1[360:] = 30.0
Tsp1[660:] = 40.0

board.send_sysex(0x02, [32])
time.sleep(0.1) #Esse sleep é necessário para dar tempo do valor da variável temperatura ser atualizado

T1 = np.ones(loops) * temperature # measured T (degC)
error_sp = np.zeros(loops)

Tsp2 = np.ones(loops) * 23.0  # set point (degC)
# T2 = np.ones(loops) * a.T2  # measured T (degC)

# Predictions

board.send_sysex(0x02, [32])
time.sleep(0.1) #Esse sleep é necessário para dar tempo do valor da variável temperatura ser atualizado

Tp = np.ones(loops) * temperature
error_eb = np.zeros(loops)
Tpl = np.ones(loops) * temperature
error_fopdt = np.zeros(loops)

# impulse tests (0 - 100%)
Q1 = np.ones(loops) * 0.0
Q2 = np.ones(loops) * 0.0
Qp = np.ones(loops) * 0.0

error = np.zeros(loops)

print('Running Main Loop. Ctrl-C to end.')
print('  Time     SP     PV     Q1   =  P   +  I  +   D')
print(('{:6.1f} {:6.2f} {:6.2f} ' +
       '{:6.2f} {:6.2f} {:6.2f} {:6.2f}').format(
           tm[0], Tsp1[0], T1[0],
           Q1[0], 0.0, 0.0, 0.0))

# Create plot
plt.figure()  # figsize=(10,7)
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
dt_error = 0.0
# Integral error
ierr = 0.0

#belbic initial param values
uMax = 3.3

ePlot = np.zeros(loops)
uPlot = np.zeros(loops)

eant = 0
iant = 0
E_dot = 0
E = 0

#Not used, because TCLAB PID is used instead of belbic PID
belbic.kp = 85
belbic.ki = 1.7
belbic.kd = 85

#For parameters original of tclab
# belbic.alpha = 0.00087
# belbic.beta = 0.00001

#For paramaters obtained in experiment regression
# belbic.alpha = 0.001
# belbic.beta = 0.0000097

#For parameters obtained in pidtuner.com
belbic.alpha = 0.001
belbic.beta = 0.00001

vi = 0
wi = 0

A = 0
O = 0

try:
    for i in range(1, loops):
        # Sleep time
        sleep_max = 1.0
        sleep = sleep_max - (time.time() - prev_time) - dt_error
        if sleep >= 1e-4:
            time.sleep(sleep-1e-4)
        else:
            print('exceeded max cycle time by ' + str(abs(sleep)) + ' sec')
            time.sleep(1e-4)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        if (sleep >= 1e-4):
            dt_error = dt-1.0+0.009
        else:
            dt_error = 0.0
        prev_time = t
        tm[i] = t - start_time

        #Update belbic sampling time
        belbic.h = dt

        # print('h: ', belbic.h)

        # Read temperatures in Kelvin

        board.send_sysex(0x02, [32])
        T1[i] = temperature
        # T2[i] = a.T2
        
        error[i] = Tsp1[i] - T1[i]

        # Simulate one time step with Energy Balance
        Tnext = odeint(heat, Tp[i-1]+273.15, [0, dt], args=(Q1[i-1],))
        Tp[i] = Tnext[1]-273.15

        # Simulate one time step with linear FOPDT model
        z = np.exp(-dt/tauP)
        Tpl[i] = (Tpl[i-1]-Tss) * z \
            + (Q1[max(0, i-int(thetaP)-1)]-Qss)*(1-z)*Kp \
            + Tss

        # Calculate PID output from TCLAB as belbic SI block
        ePlot[i] = Tsp1[i] - T1[i]
        [si, P, ierr, D] = pid(Tsp1[i], T1[i], T1[i-1], ierr, dt)

        # Original SI block from belbic
        # si, dedt, eantNew, iantNew = belbic.SI(ePlot[i], eant, iant, T1[i], T1[i - 1])
        # eant = eantNew
        # iant = iantNew

        rew = abs(ePlot[i])

        belbic.alpha = belbic.alpha * belbic.h
        belbic.beta = belbic.beta * belbic.h

        viNew, wiNew = belbic.sensory_cortex(si, rew, A, E_dot, vi, wi)
        vi = viNew
        wi = wiNew

        O, E_dot = belbic.orbifrontal_cortex(wi, si, A, O, rew)
        A, E = belbic.amygdala(vi, si, A, O, rew)

        uPlot[i] = A - O

        # print('raw signal control: ', uPlot[i])

        if (uPlot[i] >= uMax):
            uPlot[i] = uMax
        elif (uPlot[i] <= 0):
            uPlot[i] = 0
        
        
        #mapping to power
        Q1_map = (100 * uPlot[i]) / uMax
        value_Q1 = max(0, min(Q1_map, 100))
        Q1[i] = int(value_Q1 * 255) / 100
        
        Qp[i] = Q1_map

        # Start setpoint error accumulation after 1 minute (60 seconds)
        if i >= 60:
            error_eb[i] = error_eb[i-1] + abs(Tp[i]-T1[i])
            error_fopdt[i] = error_fopdt[i-1] + abs(Tpl[i]-T1[i])
            error_sp[i] = error_sp[i-1] + abs(Tsp1[i]-T1[i])

        # Write output (0-100)
        datasToWrite = []

        datasToWrite.append(12)
        datasToWrite.append(0)
        datasToWrite.append(49)
        datasToWrite.append(8)
        
        datasToWrite.append(int(Q1[i]))
        board.send_sysex(0x04, datasToWrite)

        # a.Q1(Q1[i])
        # a.Q2(0.0)

        # Print line of data
        print(('{:6.1f} {:6.2f} {:6.2f} ' +
              '{:6.2f} {:6.2f} {:6.2f} {:6.2f}').format(
                  tm[i], Tsp1[i], T1[i],
                  Q1[i], P, ierr, D))
        # plt.pause(0.05)
# Plot
    # Turn off heaters
    datasToWrite = []

    datasToWrite.append(12)
    datasToWrite.append(0)
    datasToWrite.append(49)
    datasToWrite.append(8)
    datasToWrite.append(0)
    
    board.send_sysex(0x04, datasToWrite)

    plt.clf()
    ax = plt.subplot(3, 1, 1)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tsp1, 'k--', label=r'$T_1$ set point')
    plt.ylim(25, 110)
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    ax = plt.subplot(3, 1, 2)
    ax.grid()
    plt.plot(tm, Q1, 'b-', label=r'$Q_1$')
    plt.ylabel('Heater')
    plt.legend(loc='best')
    ax = plt.subplot(3, 1, 3)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tpl, 'g-', label=r'$T_1$ linear model')
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    plt.xlabel('Time (sec)')
    plt.draw()
    plt.savefig('test_belbic.png')

    #Salvar o y (saída), sinal de controle, erro em um arquivo .txt
    save_data_belbic(T1, Qp, error)

    board.exit()

# Allow user to end loop with Ctrl-C
except KeyboardInterrupt:
    # Disconnect from ESP32
    datasToWrite = []
    datasToWrite.append(12)
    datasToWrite.append(0)
    datasToWrite.append(49)
    datasToWrite.append(8)
    datasToWrite.append(0)

    board.send_sysex(0x04, datasToWrite)

    plt.clf()
    ax = plt.subplot(3, 1, 1)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tsp1, 'k--', label=r'$T_1$ set point')
    plt.ylim(25, 110)
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    ax = plt.subplot(3, 1, 2)
    ax.grid()
    plt.plot(tm, Q1, 'b-', label=r'$Q_1$')
    plt.ylabel('Heater')
    plt.legend(loc='best')
    ax = plt.subplot(3, 1, 3)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tpl, 'g-', label=r'$T_1$ linear model')
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    plt.xlabel('Time (sec)')
    plt.draw()
    plt.savefig('test_belbic.png')

    save_data_belbic(T1, Qp, error)

    board.exit()

    print('Shutting down')

# Make sure serial connection still closes when there's an error
except:
    # Disconnect from ESP32
    datasToWrite = []
    datasToWrite.append(12)
    datasToWrite.append(0)
    datasToWrite.append(49)
    datasToWrite.append(8)
    datasToWrite.append(0)

    board.send_sysex(0x04, datasToWrite)

    plt.clf()
    ax = plt.subplot(3, 1, 1)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tsp1, 'k--', label=r'$T_1$ set point')
    plt.ylim(25, 110)
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    ax = plt.subplot(3, 1, 2)
    ax.grid()
    plt.plot(tm, Q1, 'b-', label=r'$Q_1$')
    plt.ylabel('Heater')
    plt.legend(loc='best')
    ax = plt.subplot(3, 1, 3)
    ax.grid()
    plt.plot(tm, T1, 'r.', label=r'$T_1$ measured')
    plt.plot(tm, Tpl, 'g-', label=r'$T_1$ linear model')
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    plt.xlabel('Time (sec)')
    plt.draw()
    plt.savefig('test_belbic.png')

    save_data_belbic(T1, Qp, error)

    board.exit()

    print('Error: Shutting down')
    raise
