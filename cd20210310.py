#%%
import enum
from re import T
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
phonon_dim=6
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_freq=5970.04
phonon_freq=5974.115
qubit_phonon_detuning=qubit_freq-phonon_freq

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*phonon_num
#T1 list of the system 77
t1=[11]+[77]*(phonon_num)
#T2 list of the system 104
t2=[12]+[104]*(phonon_num)

#set up the processor and compiler,qb5d97 is the qubit we play around
qb5d97_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=[0.2599,0.07],\
    rest_place=qubit_phonon_detuning,FSR=1.1)
qb5d97_compiler = hbar_compiler.HBAR_Compiler(qb5d97_processor.num_qubits,\
    qb5d97_processor.params, qb5d97_processor.pulse_dict)
qb5d97_simulation=hbar_simulation_class.Simulation(qb5d97_processor,qb5d97_compiler)


#the list of swap time between qubit and phonon for different Fock state
qb5d97_simulation.swap_time_list=[0.9620009872224587,
 0.6794930725222,
 0.5549324886188994,
 0.48034852042842774]
#%%
'''
For higher order phonon rabi
'''
pi_time_list=[]
for n in range(1):
    qb5d97_simulation.t_list=np.linspace(0.01,10,100)
    qb5d97_simulation.generate_fock_state(n)
    qb5d97_simulation.phonon_rabi_measurement()  
    pi_time_list.append(qb5d97_simulation.fit_result[-1]['swap_time'])
    qb5d97_simulation.swap_time_list=pi_time_list
#%%
'''
ramsey numbersplitting
'''
interaction_1_freq=5972.2
qb5d97_simulation.t_list=np.linspace(0.01,20,51)
y_2d_list=[]
for i in range(5):
    qb5d97_simulation.ideal_phonon_fock(i)
    qb5d97_simulation.qubit_ramsey_measurement(artificial_detuning=2.1212,
    starkshift_amp=interaction_1_freq-phonon_freq,if_fit=False)
    y_2d_list.append(qb5d97_simulation.y_array)

figure, ax = plt.subplots(figsize=(8,6))
for i in range(5):
    ax.plot(qb5d97_simulation.t_list,y_2d_list[i],label='Fock{}'.format(i))
plt.legend()
#%%
with_decay=y_2d_list
#%%
figure, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.jet(np.linspace(0.1,0.9,5))
for i in range(5):
    ax.plot(qb5d97_simulation.t_list,without_decay[i],'--',color=colors[i],label='without decay Fock{}'.format(i))
    ax.plot(qb5d97_simulation.t_list,with_decay[i],'-',color=colors[i],label='with decay Fock{}'.format(i))
plt.legend()

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
    param_probe['amplitude_starkshift']=interaction_1_freq-qubit_freq
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
fock state numbersplitting experiment 
'''
#define the parameter for drive the coherent state
param_probe={'Omega':0.01,
    'sigma': 0.5,
    'duration':30,
    'amplitude_starkshift':0}

#define the parameter for probing the qubit
y_list=[]
fock_number_list=range(1,2)
for fock_number in fock_number_list:
    qb5d97_simulation.generate_fock_state(fock_number)
    # qb5d97_simulation.ideal_phonon_fock(fock_number)
    # param_drive['detuning']=-qubit_phonon_detuning
    # qb5d97_simulation.generate_coherent_state(phonon_drive_params=param_drive)
    param_probe['amplitude_starkshift']=interaction_1_freq-qubit_freq
    qb5d97_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-0.3,
        param_probe['amplitude_starkshift']+0.05,100)
    qb5d97_simulation.spec_measurement(param_probe)
    y_list.append(qb5d97_simulation.y_array)
#plot multi peaks together
figure,ax = plt.subplots(figsize=(8,6))
for fock_number in fock_number_list:
    ax.plot(qb5d97_simulation.detuning_list ,y_list[fock_number],label='fock number={}'.format(fock_number))
figure.legend()
figure.show()
# %%
NS_fitter=hbar_fitting.fitter(qb5d97_simulation.detuning_list,y_list[3])
fit_result=NS_fitter.fit_multi_peak(4)
# %%
#coherent state generation
param_drive={'Omega':0.06*8.1,
    'sigma':0.5,
    'duration':14,
    'rotate_direction':np.pi/4,
    'detuning':-qubit_phonon_detuning
    }
qb5d97_simulation.generate_coherent_state(param_drive)
qb5d97_simulation.fit_wigner()
qt.plot_wigner(qb5d97_simulation.initial_state.ptrace(1))
result=qb5d97_simulation.alpha
abs(result)
# %%
