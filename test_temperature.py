#Pinout range of ESP32 board mapped by firmata-esp32

#define IS_PIN_DIGITAL(p)       (((p) >= 2 && (p) <= 5) || ((p) >= 13 && (p) <= 27))
#define IS_PIN_ANALOG(p)        (((p) >= 32 && (p) <= 32 + TOTAL_ANALOG_PINS))
#define IS_PIN_PWM(p)           digitalPinHasPWM(p) // all gpios in digital

from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyUSB0')
it = util.Iterator(board)
it.start()

print('✔ board ready!')


def handle_read_write(*data):
    print(data)

    datasToWrite = []

    datasToWrite.append(12)
    datasToWrite.append(0)
    datasToWrite.append(1)
    datasToWrite.append(8)
    datasToWrite.append(204)

    # value = 204

    # v = divmod(value, 127)

    # for i in range(1, v[0]):
    #     datasToWrite.append(127)

    # if (v[0] >= 1):
    #     datasToWrite.append(v[1])
    # else:
    #     datasToWrite.append(value)

    rawV = 0

    for i in data:
        rawV = rawV + i


    temperature = (rawV * (5000 / 1023)) / 10

    print(temperature)
    board.send_sysex(0x04, datasToWrite)


board.add_cmd_handler(0x02, handle_read_write)

while True:
    board.send_sysex(0x02, [32])
    time.sleep(0.01)