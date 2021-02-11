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


#fitted simulation result of swap gate time for each fock state.
swap_t_list_simulated=np.array([0.9615710875211836, 0.6793394959515644, 0.5549226661382177, 0.4804636060930446,\
     0.4294620578370378, 0.3923078531720593, 0.3639007000694595, 0.34220291598663793])
#%%

t_L=np.linspace(0.1,7,50)
# detuning_L=np.linspace(-0.1,0.3,101)
detuning_L=np.linspace(-0.1,0.3,101)

param_probe={'Omega':0.008,
    'sigma':0.01,
    'duration':30,
    }
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':7,
    'detuning':-qubit_phonon_detuning
    }

# catch_result=hbar_sequence.qubit_spec_measurement(detuning_L,test_processor,test_compiler,param_probe)
catch_result=hbar_sequence.num_split_coh_measurement(detuning_L,test_processor,test_compiler,param_drive,param_probe)
plt.plot(detuning_L,catch_result)
plt.show()
#%%
test_fitter=hbar_fitting.fitter(detuning_L,catch_result)
fit_result=test_fitter.fit_multi_peak(0.11)
reading_time=abs(1/(fit_result[0]-fit_result[1])/2)
qubit_shift=fit_result[0]

#%%
for i in range(5):
    catch_result=hbar_sequence.qubit_T2_measurement(t_L,test_processor,test_compiler,swap_time_list=swap_t_list_simulated[:i], artificial_detuning=qubit_shift)
    plt.plot(t_L,catch_result)
plt.show()
#%%
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':15,
    'detuning':-qubit_phonon_detuning-0.05
    }
Omega=0.2
steps=30
for fock_number in range(5):
    catch_result=hbar_sequence.parity_wigner_measurement(Omega,steps,test_processor,test_compiler,param_drive,reading_time=reading_time,artificial_detuning=qubit_shift,swap_time_list=swap_t_list_simulated[:fock_number])
    x=np.linspace(-Omega,Omega,steps)
    xx,yy=np.meshgrid(x,x)


# %%
#plot wigner function of generated phonon
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':15,
    'detuning':-1.55,
    'rotate_direction':np.pi/2
    }
circuit = QubitCircuit((test_processor.N))   
circuit.add_gate("XY_R_GB", targets=0,arg_value=param_drive)
final_state=hbar_sequence.run_circuit(circuit,test_processor,test_compiler)
final_state.ptrace(1)
qt.plot_wigner_fock_distribution(final_state.ptrace(1))

# %%
plt.figure(figsize=(6,6))
plt.contourf(xx, yy, catch_result*2-1,40,cmap='RdBu_r')
plt.gca().set_aspect('equal')
plt.colorbar().set_label("qubit")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# %%
reading_time
# %%
