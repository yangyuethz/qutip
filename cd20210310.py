#%%
import enum
from numpy.core.function_base import linspace
from qutip.visualization import plot_fock_distribution
from qutip.states import coherent
import hbar_compiler
import hbar_processor
import hbar_fitting
import hbar_simulation_class
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from qutip.qip.circuit import Measurement, QubitCircuit
import qutip as qt
from qutip import basis
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_fitting)
reload(hbar_simulation_class)
%matplotlib qt
#qubit dimission, we only consider g and e states here
qubit_dim=2
#phonon dimission
phonon_dim=3
#how many phonon modes we consider here
phonon_num=2
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_phonon_detuning=5968.6-5974.1

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*phonon_num
#T1 list of the system
t1=[11]+[77]*(phonon_num)
#T2 list of the system
t2=[12]+[104]*(phonon_num)
#set up the processor and compiler,qb5d97 is the qubit we play around
qb5d97_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=[0.26,0.07],\
    rest_place=qubit_phonon_detuning,FSR=1.1)
qb5d97_compiler = hbar_compiler.HBAR_Compiler(qb5d97_processor.num_qubits,\
    qb5d97_processor.params, qb5d97_processor.pulse_dict)
qb5d97_simulation=hbar_simulation_class.Simulation(qb5d97_processor,qb5d97_compiler)


#the list of swap time between qubit and phonon for different Fock state
qb5d97_simulation.swap_time_list=[0.9615913396391608,
                                    0.679369013069411,
                                    0.5548749387394517,
                                    0.48045143954729463,
                                    0.42939088344326864,
                                    0.39660104948110375,
                                    0.3688697372314131,
                                    0.34540763247883866,
                                    0.3257502876196255,0.30900063232085784]
#%%
'''
For higher order phonon rabi
'''
y_list=[]
qb5d97_simulation.t_list=np.linspace(0.01,10,100)
detuning_list=np.linspace(-0.1,0.1,21)
for detuning in detuning_list:
    qb5d97_simulation.generate_fock_state(0)
    qb5d97_simulation.phonon_rabi_measurement(detuning=detuning)  
    y_list.append(qb5d97_simulation.y_array)

#%%
xx,yy=np.meshgrid(detuning_list,qb5d97_simulation.t_list)
plt.pcolormesh(xx,yy,np.array(y_list).transpose())
plt.ylabel('(us)')
plt.xlabel('detuning(MHz)')
#%%
'''
This part is for calibrating probe amplitude or pulse length,
'''
param_probe={'Omega':0.015,
    'sigma': 0.5,
    'duration':15,
    'amplitude_starkshift':0}

y_list=[]
sweep_list=np.linspace(0.005,0.02,10)
for sweep_data in sweep_list:
    param_probe['Omega']=sweep_data
    param_probe['duration']=30
    qb5d97_simulation.ideal_phonon_fock(0)
    param_probe['amplitude_starkshift']=5973.3-5968.6
    qb5d97_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-0.2,
        param_probe['amplitude_starkshift']+0.2,41)
    qb5d97_simulation.spec_measurement(param_probe)
    y_list.append(qb5d97_simulation.y_array)

figure,ax = plt.subplots(figsize=(8,6))
for i,sweep_data in enumerate(sweep_list):
    ax.plot(qb5d97_simulation.detuning_list ,y_list[i],label='probe omega={}MHz'.format(sweep_data))
figure.legend()
figure.show()

#%%
'''
numbersplitting experiment 
'''
#define the parameter for drive the coherent state

param_drive={'Omega':0.1,
    'sigma':0.2,
    'duration':10,
    'rotate_direction':np.pi/4
    }

param_probe={'Omega':0.01,
    'sigma': 0.5,
    'duration':30,
    'amplitude_starkshift':0}

#define the parameter for probing the qubit
y_list=[]
fock_number_list=range(4)
for fock_number in fock_number_list:
    qb5d97_simulation.generate_fock_state(fock_number)
    # qb5d97_simulation.ideal_phonon_fock(fock_number)
    # param_drive['detuning']=-qubit_phonon_detuning
    # qb5d97_simulation.generate_coherent_state(phonon_drive_params=param_drive)
    param_probe['amplitude_starkshift']=5972.3-5968.6
    qb5d97_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-0.6,
        param_probe['amplitude_starkshift']+0.2,100)
    qb5d97_simulation.spec_measurement(param_probe)
    y_list.append(qb5d97_simulation.y_array)
figure,ax = plt.subplots(figsize=(8,6))
for fock_number in fock_number_list:
    ax.plot(qb5d97_simulation.detuning_list ,y_list[fock_number],label='fock number={}'.format(fock_number))
figure.legend()
figure.show()
# %%
param_probe={'Omega':0,
    'sigma': 0.5,
    'duration':14,
    'amplitude_starkshift':0}
param_probe['amplitude_starkshift']=5973.3-5968.6
qb5d97_simulation.t_list=np.linspace(14,16,100)
y_list=[]
fock_number_list=range(4)
for fock_number in fock_number_list:
    qb5d97_simulation.generate_fock_state(fock_number)
    qb5d97_simulation.qubit_shift_wait(param_probe)
    y_list.append(qb5d97_simulation.y_array)
figure,ax = plt.subplots(figsize=(8,6))
for fock_number in fock_number_list:
    ax.plot(qb5d97_simulation.t_list ,y_list[fock_number],label='fock number={}'.format(fock_number))
figure.legend()
# %%
