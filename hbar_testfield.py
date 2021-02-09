#%%
import hbar_compiler
import hbar_processor
import hbar_sequence
import hbar_fitting
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
reload(hbar_compiler)
reload(hbar_processor)
reload(hbar_sequence)
reload(hbar_fitting)


qubit_dim=2
phonon_dim=5
phonon_num=1
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
t1=[16]+[60]*(phonon_num)
t2=[17]+[112]*(phonon_num)
test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,rest_place=1)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)

#fitted simulation result of swap gate time for each fock state.
swap_t_list_simulated=np.array([0.9615710875211836, 0.6793394959515644, 0.5549226661382177, 0.4804636060930446,\
     0.4294620578370378, 0.3923078531720593, 0.3639007000694595, 0.34220291598663793])
#%%
t_L=np.linspace(0.1,5,100)
detuning_L=np.linspace(-0.1,0.5,121)

param_probe={'Omega':0.00625,
    'sigma':0.01,
    'duration':40,
    }
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':4,
    'detuning':-1
    }

# catch_result=hbar_sequence.qubit_T2_measurement(t_L,test_processor,test_compiler,swap_time_list=swap_t_list_simulated[:4], artifical_detuning=0.06)
# catch_result=hbar_sequence.qubit_spec_measurement(detuning_L,test_processor,test_compiler,param_probe)
catch_result=hbar_sequence.num_split_coh_measurement(detuning_L,test_processor,test_compiler,param_drive,param_probe)

plt.plot(t_L,catch_result)
# plt.plot(detuning_L,catch_result)

# %%
test_fitter=hbar_fitting.fitter(detuning_L,catch_result)
fit_result=test_fitter.fit_multi_peak()
# %%
test_processor.plot_pulses()
# %%
fit_result
# %%
