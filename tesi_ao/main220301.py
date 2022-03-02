from tesi_ao import sandbox, package_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate.interpolate import interp1d
'''
all the following is related to actuator #63
'''
wyko, bmc = sandbox.create_devices()
actuator_list = [63]

mcl0, cplm0, cpla0 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl0.fits', 'prova/63cpl0.fits', actuator_list)
mcl1, cplm1, cpla1 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl1.fits', 'prova/63cpl1.fits', actuator_list)
mcl2, cplm2, cpla2 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl2.fits', 'prova/63cpl2.fits', actuator_list)
mcl3, cplm3, cpla3 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl3.fits', 'prova/63cpl3.fits', actuator_list)
mcl4, cplm4, cpla4 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl4.fits', 'prova/63cpl4.fits', actuator_list)
mcl5, cplm5, cpla5 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl5.fits', 'prova/63cpl5.fits', actuator_list)
mcl6, cplm6, cpla6 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl6.fits', 'prova/63cpl6.fits', actuator_list)
mcl7, cplm7, cpla7 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl7.fits', 'prova/63cpl7.fits', actuator_list)
mcl8, cplm8, cpla8 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl8.fits', 'prova/63cpl8.fits', actuator_list)
mcl9, cplm9, cpla9 = sandbox.main_calibration(
    wyko, bmc, 'prova/63mcl9.fits', 'prova/63cpl9.fits', actuator_list)
# 10 repeated measures for act#63

deflections0 = mcl0._deflection[0]
deflections1 = mcl1._deflection[0]
deflections2 = mcl2._deflection[0]
deflections3 = mcl3._deflection[0]
deflections4 = mcl4._deflection[0]
deflections5 = mcl5._deflection[0]
deflections6 = mcl6._deflection[0]
deflections7 = mcl7._deflection[0]
deflections8 = mcl8._deflection[0]
deflections9 = mcl9._deflection[0]

deflection_list = [deflections1, deflections2, deflections3, deflections4,
                   deflections5, deflections6, deflections7, deflections8, deflections9]


cmds = mcl0._cmd_vector[0]


plt.figure(1)
plt.clf()
plt.ion()
plt.grid()
plt.title('Actuator #%d responce' % actuator_list[0])
plt.plot(deflections0, cmds, 'ko', label='data')
plt.xlabel('Deflections [m]')
plt.ylabel('Commands [au]')
plt.legend(loc='best')

plt.figure(2)
plt.clf()
plt.ion()
plt.title('reproducibility act #63')
j = 0
for i in np.arange(len(deflection_list)):
    j += 1
    plt.plot(cmds, deflection_list[i] -
             deflections0, '.-', label='meas%d' % j)
plt.legend(loc='best')
plt.grid()
plt.xlabel('Commands [au]')
plt.ylabel('Displacement difference [m]')


plt.figure(3)
plt.clf()
plt.ion()
plt.title('response act #63')
for i in np.arange(len(deflection_list)):
    plt.plot(cmds, deflection_list[i], '.-', label='meas%d' % i)
plt.plot(cmds, deflections0, '.-', label='meas0')
plt.legend(loc='best')
plt.grid()
plt.xlabel('Commands [au]')
plt.ylabel('Displacement [m]')

plt.figure(4)  # interpolation plotting
plt.clf()
plt.ion()
f0 = mcl0._finter[0]
def0_span = np.linspace(min(deflections0), max(deflections0), 1000)

plt.title('response act #63')
plt.plot(f0(def0_span), def0_span, '-', label='interpol0')
plt.plot(cmds, deflections0, 'o', label='data0')
plt.xlabel('Commands [au]')
plt.ylabel('Displacement [m]')

plt.figure(5)  # position error
plt.clf()
plt.ion()
plt.title('position error act #63')
plt.xlabel('$position [m]$')
plt.ylabel('$p_{obs} - p_{exp}$')
