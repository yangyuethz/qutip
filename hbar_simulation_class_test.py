#%%
import hbar_compiler
import hbar_processor
import hbar_sequence
import hbar_fitting
import hbar_sequence_class
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from qutip.qip.circuit import Measurement, QubitCircuit
import qutip as qt
from  qutip.parallel import parallel_map, parfor 
from qutip import basis
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_sequence)
reload(hbar_fitting)
reload(hbar_sequence_class)
%matplotlib qt
#qubit dimission, we only consider g and e states here
qubit_dim=2   
#phonon dimission
phonon_dim=15
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon
qubit_phonon_detuning=1.5
#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
#T1 list of the system
t1=[16]+[60]*(phonon_num)
#T2 list of the system
t2=[17]+[112]*(phonon_num)
#set up the processor and compiler
test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,\
    rest_place=qubit_phonon_detuning)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)

# %%
simulation_test=hbar_sequence_class.Simulation(test_processor,test_compiler)
simulation_test.swap_time_list=np.array([0.9615745472779998, 0.6793420222651693, 0.5549538733587259,
 0.48038367226189926, 0.4292531978245541, 0.39238362592764875, 0.36387759767336825, 0.34112748801301906])

# %%
simulation_test.detuning_list=np.linspace(-1.7,-1.4,100)
param_probe={'Omega':0.2,
    'sigma':0.01,
    'duration':5,
    }
simulation_test.spec_measurement(param_probe,readout_type='read phonon')
# %%
simulation_test.fitter.fit_single_peak()
# %%
