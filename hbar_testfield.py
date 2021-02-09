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
t_L=np.linspace(0.1,10,100)
detuning_L=np.linspace(-0.2,0.5,71)
param1={'Omega':0.0125,
    'sigma':0.2,
    'duration':15,
    }
param2={'Omega':0.3,
    'sigma':0.2,
    'duration':30,
    'detuning':-1
    }
catch_result=hbar_sequence.qubit_T2_measurement(t_L,test_processor,test_compiler,artifical_detuning=1)

plt.plot(t_L,catch_result)

# %%
