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
phonon_dim=5
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
qb5d97_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=[0.07,0.07],\
    rest_place=qubit_phonon_detuning,FSR=1.1,coupling='dispersive H')
qb5d97_compiler = hbar_compiler.HBAR_Compiler(qb5d97_processor.num_qubits,\
    qb5d97_processor.params, qb5d97_processor.pulse_dict)
qb5d97_simulation=hbar_simulation_class.Simulation(qb5d97_processor,qb5d97_compiler)

#the list of swap time between qubit and phonon for different Fock state
qb5d97_simulation.swap_time_list=[0.9620009872224587,
 0.6794930725222,
 0.5549324886188994,
 0.48034852042842774]

# %%
# interaction_1_freq=5972.2
# qb5d97_simulation.t_list=np.linspace(0.01,20,50)
# y_2d_list=[]
for i in range(4,5):
    qb5d97_simulation.ideal_phonon_fock(i)
    qb5d97_simulation.qubit_ramsey_measurement(artificial_detuning=0,
    starkshift_amp=qubit_phonon_detuning,if_fit=False)
    y_2d_list.append(qb5d97_simulation.y_array)

figure, ax = plt.subplots(figsize=(8,6))
for i in range(5):
    ax.plot(qb5d97_simulation.t_list,y_2d_list[i],label='Fock{}'.format(i))
plt.legend()
# %%
perfect_dipsersive_H_list=y_2d_list
# %%
