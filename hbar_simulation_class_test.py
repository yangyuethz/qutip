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
phonon_dim=30
#how many phonon modes we consider here
phonon_num=1
#the frequency difference between qubit and phonon
qubit_phonon_detuning=1.5
#dimission of the system, qubit dimission + phonons dimission
dims=[qubit_dim]+[phonon_dim]*(phonon_num)
#T1 list of the system
t1=[16]+[60]*(phonon_num)
# t1=[1000]*(1+phonon_num)
#T2 list of the system
t2=[17]+[112]*(phonon_num)
# t2=[1000]*(1+phonon_num)
#set up the processor and compiler
test_processor=hbar_processor.HBAR_processor((phonon_num+1),t1,t2,dims,g=0.26,\
    rest_place=qubit_phonon_detuning)
test_compiler = hbar_compiler.HBAR_Compiler(test_processor.num_qubits,\
    test_processor.params, test_processor.pulse_dict)
simulation_test=hbar_simulation_class.Simulation(test_processor,test_compiler)
#set up parameters
simulation_test.swap_time_list=np.array([0.9615745472779998, 0.6793420222651693, 0.5549538733587259,
 0.48038367226189926, 0.4292531978245541, 0.39238362592764875, 0.36387759767336825, 0.34112748801301906])

phonon_freq=-1.538
reading_time=6.942878985988443
qubit_freq=0.056123727087823316
simulation_test.reading_time=reading_time
simulation_test.artificial_detuning=qubit_freq
simulation_test.t_list=np.linspace(0.1,30,60)

param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':10,
    'detuning':phonon_freq,
    'rotate_direction':0
    }
dis_entangle_omega=0.1/np.sqrt(3)
param_probe={'Omega':dis_entangle_omega,
    'sigma':0.01,
    'duration':1/dis_entangle_omega/4,
    'detuning':qubit_freq
    }
simulation_test.phonon_drive_params=param_drive
simulation_test.qubit_probe_params=param_probe
#%%
catched_data=simulation_test.wigner_measurement_2D(steps=10)
#%%
x=np.linspace(-1.76,1.76,10)
xx,yy=np.meshgrid(x,x)
plt.figure(figsize=(6,6))
plt.contourf(xx, yy, catched_data,40)
plt.gca().set_aspect('equal')
plt.colorbar().set_label("qubit")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#%% test cat state
simulation_test.generate_cat()
#%% find phonon freq
'''
drive on phonon freq and readout phonon expected freq
'''
simulation_test.ideal_qubit_state(0)
simulation_test.detuning_list=np.linspace(-qubit_phonon_detuning-0.15,-qubit_phonon_detuning+0.1,43)
simulation_test.spec_measurement(param_drive,readout_type='read phonon')
phonon_freq=simulation_test.fitter.fit_single_peak()['w0']
param_drive['detuning']=phonon_freq
#%% 
'''
find qubit freq and number splitting freq
the number splitting and qubit freq found by this way works not that good in parity measurement
'''
simulation_test.detuning_list=np.linspace(-0.05,0.1,30)
simulation_test.ideal_phonon_fock(0)
simulation_test.spec_measurement(param_probe)
qubit_freq=simulation_test.fitter.fit_single_peak()['w0']
simulation_test.ideal_phonon_fock(1)
simulation_test.detuning_list=np.linspace(-0.,0.25,30)
simulation_test.spec_measurement(param_probe)
number_splitting_freq=simulation_test.fitter.fit_single_peak()['w0']-qubit_freq

#%% 
'''
using qubit ramsey to find qubit freq and number splitting 
normally the measured result by ramsey is better for parity measurement
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

simulation_test.t_list=np.linspace(0.1,reading_time,2)
parity_list=[]
for i in range(8):
    simulation_test.ideal_phonon_fock(i)
    simulation_test.qubit_ramsey_measurement(artificial_detuning=qubit_freq,fit=False)
    parity_list.append(simulation_test.y_array[-1]*2-1)
fig , ax =plt.subplots()
ax.scatter(range(8),parity_list)
plt.title('parity measurement',fontsize=25)
plt.xlabel('phonon number',fontsize=18)
plt.ylabel("measured parity",fontsize=18)
plt.show()
#%%
'''
test parity measurement of wigner function, simulated displacement
'''
simulation_test.ideal_phonon_fock(2)
simulation_test.wigner_measurement_1D(param_drive,second_pulse_flip=True)
y_array_stored=simulation_test.y_array
simulation_test.ideal_phonon_fock(2)
simulation_test.wigner_measurement_1D(param_drive,second_pulse_flip=False)
#plot offset
fig , ax =plt.subplots()
ax.plot(simulation_test.x_array, y_array_stored+simulation_test.y_array-1)
plt.show()
#plot caliberated parity
fig , ax =plt.subplots()
ax.plot(simulation_test.x_array,(-( y_array_stored+simulation_test.y_array-1)/2+simulation_test.y_array)*2-1)
plt.show()

# %%
'''
test parity measurement of wigner function, ideal displacement
'''
simulation_test.ideal_phonon_fock(2)
simulation_test.wigner_measurement_1D(param_drive,second_pulse_flip=True,displacement_type='ideal',set_alpha_range=5)
y_array_stored=simulation_test.y_array
simulation_test.ideal_phonon_fock(2)
simulation_test.wigner_measurement_1D(param_drive,second_pulse_flip=False,displacement_type='ideal',set_alpha_range=5)
#%%
#plot offset
fig , ax =plt.subplots()
ax.plot(simulation_test.x_array, y_array_stored+simulation_test.y_array-1)
plt.show()

#plot caliberated parity
fig , ax =plt.subplots()
ax.plot(simulation_test.x_array,(+( y_array_stored+simulation_test.y_array-1)/2+simulation_test.y_array)*2-1)
plt.show()

# %%
simulation_test.ideal_coherent_state(5)
simulation_test.qubit_ramsey_measurement()
# %%

