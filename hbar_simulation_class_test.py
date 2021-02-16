#%%
import hbar_compiler
import hbar_processor
import hbar_sequence
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
reload(hbar_sequence)
reload(hbar_fitting)
reload(hbar_simulation_class)
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

simulation_test=hbar_simulation_class.Simulation(test_processor,test_compiler)
simulation_test.swap_time_list=np.array([0.9615745472779998, 0.6793420222651693, 0.5549538733587259,
 0.48038367226189926, 0.4292531978245541, 0.39238362592764875, 0.36387759767336825, 0.34112748801301906])

param_drive={'Omega':0.25,
    'sigma':0.2,
    'duration':10,
    'detuning':-1.521,
    'rotate_direction':0
    }
param_probe={'Omega':0.025,
    'sigma':0.01,
    'duration':10,
    }


#%% find phonon freq
simulation_test.detuning_list=np.linspace(-qubit_phonon_detuning-0.75,-qubit_phonon_detuning+0.05,63)
simulation_test.spec_measurement(param_drive,readout_type='read phonon')
phonon_freq=simulation_test.fitter.fit_single_peak()['w0']
param_drive['detuning']=phonon_freq
#%% 
'''
find qubit freq and number splitting freq
the number splitting and qubit freq found by this way works not that good in parity measurement
'''
simulation_test.detuning_list=np.linspace(-0.2,0.5,120)
simulation_test.ideal_phonon_fock(0)
simulation_test.spec_measurement(param_probe)
qubit_freq=simulation_test.fitter.fit_single_peak()['w0']
simulation_test.ideal_phonon_fock(1)
simulation_test.spec_measurement(param_probe)
number_splitting_freq=simulation_test.fitter.fit_single_peak()['w0']-qubit_freq

#%% 
'''
using qubit ramsey to find qubit freq and number splitting 
'''
simulation_test.t_list=np.linspace(0.1,10,100)
detuning_list=[]
for i in range(8):
    simulation_test.ideal_phonon_fock(i)
    simulation_test.qubit_ramsey_measurement(artificial_detuning=-0.5)
    detuning_list.append(simulation_test.fit_result[-1]['delta'])

# %%
'''
test parity measurement 
'''
# reading_time=1/np.polyfit(range(8),detuning_list,1)[0]/2
# qubit_freq=np.polyfit(range(8),detuning_list,1)[1]-0.5


simulation_test.t_list=np.linspace(0.1,reading_time,100)
parity_list=[]
for i in range(8):
    simulation_test.ideal_phonon_fock(i)
    simulation_test.qubit_ramsey_measurement(artificial_detuning=qubit_freq)
    parity_list.append(simulation_test.y_array[-1]*2-1)
fig , ax =plt.subplots()
ax.scatter(range(8),parity_list)
plt.title('parity measurement',fontsize=25)
plt.xlabel('phonon number',fontsize=18)
plt.ylabel("measured parity",fontsize=18)
plt.show()
#%%
'''
test parity measurement of wigner function
'''
reading_time=6.942878985988443
qubit_freq=0.056123727087823316
simulation_test.reading_time=reading_time
simulation_test.artificial_detuning=qubit_freq
simulation_test.ideal_phonon_fock(2)
simulation_test.wigner_measurement_1D(param_drive,second_pulse_flip=True)
# %%
