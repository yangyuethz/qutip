#%%
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
phonon_dim=15
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon (qubit minus phonon)
qubit_phonon_detuning=5973-5974.1

#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
#T1 list of the system
t1=[14.3]+[85]*(phonon_num)
#T2 list of the system
t2=[15.8]+[112]*(phonon_num)
#set up the processor and compiler,qb5d97 is the qubit we play around
qb5d97_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,\
    rest_place=qubit_phonon_detuning)
qb5d97_compiler = hbar_compiler.HBAR_Compiler(qb5d97_processor.num_qubits,\
    qb5d97_processor.params, qb5d97_processor.pulse_dict)
qb5d97_simulation=hbar_simulation_class.Simulation(qb5d97_processor,qb5d97_compiler)

param_drive={'Omega':0.1,
    'sigma':0.2,
    'duration':10,
    'rotate_direction':np.pi/4
    }

# %%
qb5d97_simulation.ideal_qubit_state(0)
# qb5d97_simulation.detuning_list=np.linspace(-qubit_phonon_detuning-0.15,-qubit_phonon_detuning+0.1,43)
# qb5d97_simulation.spec_measurement(param_drive,readout_type='read phonon')
# phonon_freq=qb5d97_simulation.fitter.fit_single_peak()['w0']
# phonon_freq=1.14202
param_drive['detuning']=-qubit_phonon_detuning
# %%
# qb5d97_simulation.generate_coherent_state(param_drive)
qb5d97_simulation.ideal_coherent_state(2)
qt.plot_fock_distribution(qb5d97_simulation.initial_state.ptrace(0))
qt.plot_fock_distribution(qb5d97_simulation.initial_state.ptrace(1))
# qb5d97_simulation.fit_wigner()
# qt.plot_wigner(qb5d97_simulation.initial_state.ptrace(1))

# %%
qb5d97_simulation.ideal_coherent_state(2)
param_probe={'Omega':0.025,
    'sigma':0.5,
    'duration':10.5
    }
qb5d97_simulation.detuning_list=np.linspace(-1,+0.5,101)
qb5d97_simulation.spec_measurement(param_probe)
# %%
qb5d97_simulation.qubit_probe_params=param_probe
qb5d97_simulation.t_list=np.linspace(0.1,20,100)
qb5d97_simulation.qubit_rabi_measurement()
# %%
