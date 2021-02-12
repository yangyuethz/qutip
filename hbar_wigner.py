#%%
from qutip.states import coherent
import qutip
from qutip.operators import num
from qutip.expect import expect
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
from qutip import basis
from tqdm import tqdm
from qutip.solver import Options
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
#%%
#run qubit spec phonon number splitting to see where the qubit is and the number splitting chi
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
#ploting and fit qubit spec with phonon driving to coherent state 
test_fitter=hbar_fitting.fitter(detuning_L,qubitspec_result)
fit_result=test_fitter.fit_multi_peak(0.15)
reading_time=abs(1/(fit_result[0][0]-fit_result[0][1])/2)
qubit_shift=fit_result[0][0]
print(fit_result)
# %%
#run qubit phonon rabi oscillation up to fock 8
t_L=np.linspace(0.1,10,100)
swap_time_list_simulated=[]
for i in range(8):
    phonon_rabi_result=hbar_sequence.phonon_rabi_measurement(t_L,test_processor,test_compiler,swap_time_list_simulated)
    test_fitter=hbar_fitting.fitter(t_L,phonon_rabi_result)
    swap_time_fitted=test_fitter.fit_phonon_rabi()['swap_time']
    swap_time_list_simulated.append(swap_time_fitted)
# %%
#check the generated fock state fidelity
fock_state_generated_list=[]
prepared_state=basis( test_processor.dims, [0]+[0]*(test_processor.N-1))
for swap_t in swap_time_list_simulated:
    circuit = QubitCircuit((test_processor.N))
    circuit.add_gate("X_R", targets=0)
    circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
    prepared_state=hbar_sequence.run_circuit(circuit,test_processor,test_compiler,init_state=prepared_state)
    fock_state_generated_list.append(prepared_state)

fock_state_prepared_fidelity=[]
fock_num_list=[]
for i, state in enumerate(fock_state_generated_list):
    fock_state_prepared_fidelity.append(expect(state.ptrace(1),basis(phonon_dim,i+1)))
    fock_num_list.append(i+1)
plt.scatter(fock_num_list, fock_state_prepared_fidelity)
print(np.array([fock_num_list,fock_state_prepared_fidelity]).transpose())
#%%
artificial_detuning=0.128-0.085
result_list_2D=[]
t_L=np.linspace(0.01,reading_time,100)
for n in range(8):
    result_list_1D=[]

    plt.ion()
    y_array=np.zeros(len(t_L))
    figure, ax = plt.subplots(figsize=(8,6))
    line1, = ax.plot(t_L, y_array)
    ax.set_ylim((0,1))

    for i,t in enumerate(t_L):
        circuit = QubitCircuit((test_processor.N))
        circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
        circuit.add_gate('Wait',targets=0,arg_value=t)
        circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,'rotate_direction':2*np.pi*artificial_detuning*t})
        test_processor.load_circuit(circuit, compiler=test_compiler)
        option=Options()
        option.store_final_state=True
        option.store_states=False
        result=test_processor.run_state(init_state =basis(test_processor.dims, [0]+[n]*(test_processor.N-1)),options=option)
        result_list_1D.append(expect(num(test_processor.dims[0]),result.final_state.ptrace(0)))
        y_array[i]=result_list_1D[-1]
        line1.set_ydata(y_array)
        figure.canvas.draw()
        figure.canvas.flush_events()
    plt.plot(t_L,result_list_1D,label='fock state %d:' %(n))
    result_list_2D.append(result_list_1D)
plt.show()
fit_delta_list=[]
for result_list_1D in result_list_2D:
    test_fitter=hbar_fitting.fitter(t_L,result_list_1D)
    fit_delta_list.append(test_fitter.fit_T2()['delta'])
#%%
plt.scatter(range(8),fit_delta_list)
reading_time=1/np.polyfit(range(8),fit_delta_list,1)[0]/2
print(reading_time)

#%%
#run ramsey phonon number splitting by simulation generation
t_L=np.linspace(0.1,reading_time,50)
parity_measured_state_list=[]
for i in range(8):
    catch_result=hbar_sequence.qubit_T2_measurement(t_L,test_processor,test_compiler,swap_time_list=swap_time_list_simulated[:i], artificial_detuning=qubit_shift)
    plt.plot(t_L,catch_result,label='fock state %d:' %(i+1))
    parity_measured_state_list.append((catch_result[-1]-0.5)*2)
plt.legend()
plt.show()
print(parity_measured_state_list)
# %%
# do phonon spec to see where the phonon actual freq is
detuning_L=np.linspace(-qubit_phonon_detuning-0.2,-qubit_phonon_detuning+0.1,50)
phonon_num_list=[]
for detuning in tqdm(detuning_L):
    param_drive={'Omega':0.2,
        'sigma':0.2,
        'duration':15,
        'detuning':detuning,
        'rotate_direction':np.pi/2
        }
    circuit = QubitCircuit((test_processor.N))   
    circuit.add_gate("XY_R_GB", targets=0,arg_value=param_drive)
    final_state=hbar_sequence.run_circuit(circuit,test_processor,test_compiler)
    phonon_num_list.append(expect(num(phonon_dim),final_state.ptrace(1)))
plt.plot(detuning_L,phonon_num_list)
# %%
test_fitter=hbar_fitting.fitter(detuning_L,phonon_num_list)
fit_result=test_fitter.fit_single_peak()
phonon_freq=fit_result['w0']
print(phonon_freq)
# %%
param_drive={'Omega':0.2,
    'sigma':0.2,
    'duration':15,
    'detuning':phonon_freq,
    'rotate_direction':np.pi/2
    }
circuit = QubitCircuit((test_processor.N))   
circuit.add_gate("XY_R_GB", targets=0,arg_value=param_drive)
final_state=hbar_sequence.run_circuit(circuit,test_processor,test_compiler)
qt.plot_wigner_fock_distribution(final_state.ptrace(1))
# %%
wigner_array=np.linspace(-5,5,101)
wigner_2D=qt.wigner(final_state.ptrace(1),wigner_array,wigner_array)
position=np.where(wigner_2D==np.amax(wigner_2D))
alpha=(1j*wigner_array[position[0]]+wigner_array[position[1]])[0]/np.sqrt(2)
print(alpha)
print(expect(final_state.ptrace(1),qt.coherent(phonon_dim,alpha)))

# %%
import numpy as np
import time
import matplotlib.pyplot as plt
%matplotlib qt
x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.ion()

figure, ax = plt.subplots(figsize=(8,6))
line1, = ax.plot(x, y)

plt.title("Dynamic Plot of sinx",fontsize=25)

plt.xlabel("X",fontsize=18)
plt.ylabel("sinX",fontsize=18)

for p in range(100):
    updated_y = np.cos(x-0.05*p)
    
    line1.set_xdata(x)
    line1.set_ydata(updated_y)
    
    figure.canvas.draw()
    
    figure.canvas.flush_events()
    time.sleep(0.1)

# %%
