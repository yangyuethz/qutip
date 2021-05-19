#%%
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
phonon_dim=5
#how many phonon modes we consider here
phonon_num=2
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_phonon_detuning=5984.0-5974.1

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]+[2]
#T1 list of the system
t1=[11.4]+[85]*(phonon_num)
#T2 list of the system
t2=[11.8]+[140]*(phonon_num)
#set up the processor and compiler,qb5d97 is the qubit we play around
qb5d97_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,\
    rest_place=qubit_phonon_detuning,FSR=12.5)
qb5d97_compiler = hbar_compiler.HBAR_Compiler(qb5d97_processor.num_qubits,\
    qb5d97_processor.params, qb5d97_processor.pulse_dict)
qb5d97_simulation=hbar_simulation_class.Simulation(qb5d97_processor,qb5d97_compiler)

param_drive={'Omega':0.1,
    'sigma':0.2,
    'duration':10,
    'rotate_direction':np.pi/4
    }
param_probe={'Omega':0.015,
    'sigma': 0.5,
    'duration':14,
    'amplitude_starkshift':0}
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
#higher order phonon rabi
for i in range(10):
    qb5d97_simulation.generate_fock_state(i)
    qb5d97_simulation.phonon_rabi_measurement()

                                    
#%%
qb5d97_simulation.t_list=np.linspace(0.01,20,50)
qb5d97_simulation.ideal_phonon_fock(0)

param_probe['Omega']=0.015
qb5d97_simulation.qubit_rabi_measurement(param_probe)
param_probe['duration']=qb5d97_simulation.fit_result[-1]['swap_time']


#%%
for fock_number in range(4):
    qb5d97_simulation.generate_fock_state(fock_number)
    print(qb5d97_simulation.initial_state)
    param_probe['amplitude_starkshift']=5972.2-5984
    qb5d97_simulation.detuning_list=np.linspace(
        param_probe['amplitude_starkshift']-0.8,
        param_probe['amplitude_starkshift']+0.2,100)
    qb5d97_simulation.spec_measurement(param_probe)

#%%
for i in range(10):
    qb5d97_simulation.generate_fock_state(i)
# %%
qb5d97_simulation.generate_fock_state(2)
# %%
plot_fock_distribution(qb5d97_simulation.initial_state.ptrace(1))
# %%
qt.expect(qb5d97_simulation.initial_state.ptrace(1),qt.fock(qb5d97_processor.dims[1],11))
# %%
qb5d97_simulation.generate_fock_state(10)
plot_fock_distribution(qb5d97_simulation.initial_state.ptrace(1))
# %%
qb5d97_simulation.generate_fock_state(10)
qb5d97_simulation.spec_measurement()