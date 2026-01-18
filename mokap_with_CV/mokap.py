#!/usr/bin/env python
import sys
from mokap.core import MultiCam
from mokap.gui import QApplication, MainWindow
from mokap.core.hardware import MQTTLogger, MQTTWriter, MQTTclient
from mokap.CV.CV import CVCalc

mqttclient = MQTTclient()
mqttin = MQTTLogger(mqttclient)
mqttout = MQTTWriter(mqttclient)
mc = MultiCam(config='./config.yaml', triggered=True, silent=False, mqttlogger=mqttin)
cv = []
cv.append(CVCalc(cvinit='./CV_init.pkl',minit='./measurement.csv',mqttwriter=mqttout,multicam=mc,camidx=0))

# Example:
# Set some default parameters for all cameras at once

mc.exposure = 5000
mc.framerate = 50
mc.gamma = 1.0
mc.blacks = 0
mc.gain = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)

    if mc.nb_cameras == 0:
        exit()

    main_window = MainWindow(mc,cv)
    main_window.show()

    sys.exit(app.exec())