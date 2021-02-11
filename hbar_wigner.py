#%%
import hbar_compiler
import hbar_processor
import hbar_sequence
import hbar_fitting
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from qutip.qip.circuit import Measurement, QubitCircuit
import qutip as qt
from  qutip.parallel import parallel_map, parfor 
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_sequence)
reload(hbar_fitting)
#%%
qubit_dim=2
phonon_dim=15
phonon_num=1
qubit_phonon_detuning=1.5
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)
test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,rest_place=qubit_phonon_detuning)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)
t_L=np.linspace(0.1,7,50)
detuning_L=np.linspace(-0.1,0.3,101)
swap_time_list_simulated=np.array([0.9615745472779998, 0.6793420222651693, 0.5549538733587259,
 0.48038367226189926, 0.4292531978245541, 0.39238362592764875, 0.36387759767336825, 0.34112748801301906])

param_probe={'Omega':0.01,
    'sigma':0.01,
    'duration':25,
    }
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':5,
    'detuning':-qubit_phonon_detuning
    }
qubitspec_result=hbar_sequence.num_split_coh_measurement(detuning_L,test_processor,test_compiler,param_drive,param_probe)
plt.plot(detuning_L,qubitspec_result)
plt.show()
# %%
test_fitter=hbar_fitting.fitter(detuning_L,qubitspec_result)
fit_result=test_fitter.fit_multi_peak(0.15)
reading_time=abs(1/(fit_result[0]-fit_result[1])/2)
qubit_shift=fit_result[0]
# %%
t_L=np.linspace(0.1,10,100)
swap_time_list_simulated=[]
for i in range(8):
    phonon_rabi_result=hbar_sequence.phonon_rabi_measurement(t_L,test_processor,test_compiler,swap_time_list_simulated)
    test_fitter=hbar_fitting.fitter(t_L,phonon_rabi_result)
    swap_time_fitted=test_fitter.fit_phonon_rabi()['swap_time']
    swap_time_list_simulated.append(swap_time_fitted)

# %%
