from os import error
from qutip.tensor import tensor
from qutip.states import coherent, fock 
from numpy.core.records import array
from qutip.expect import expect
import numpy as np
from qutip.solver import Options
from qutip import basis
from qutip.qip.circuit import Measurement, QubitCircuit
from qutip.solver import Options
from qutip.operators import num, phase
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import hbar_fitting

class Simulation():
    '''
    Setting the simulated experiment
    '''
    def __init__(self,processor,compiler,t_list=np.linspace(0.01,10,100),
        detuning_list=np.linspace(-0.3,1,100),swap_time_list=[],artificial_detuning=0,
        reading_time=None,initial_state=None):
        self.qubit_probe_params={}
        self.phonon_drive_params={}
        self.processor=processor
        self.compiler=compiler
        self.t_list=t_list
        self.detuning_list=detuning_list
        self.swap_time_list=swap_time_list
        self.artificial_detuning=artificial_detuning
        self.reading_time=reading_time
        self.fit_result=[]
        if not initial_state:
            self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        else:
            self.initial_state=initial_state

    def run_circuit(self,circuit):
        self.processor.load_circuit(circuit, compiler=self.compiler)
        option=Options()
        option.store_final_state=True
        option.store_states=False
        result=self.processor.run_state(init_state =self.initial_state,options=option)
        state=result.final_state
        return state 

    def set_up_1D_experiment(self,title='simulaiton',xlabel='t(us)'):
        '''
        set up experiment for 1D system. 
        self.final_state_list is the list catch the density matrix of final state
        self.y_array is the list catch the qubit excited population of final state
        and then set up to plot the figure

        '''
        self.final_state_list=[]
        self.y_array=np.zeros(len(self.x_array))
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(8,6))
        self.line, = self.ax.plot(self.x_array, self.y_array)
        plt.title(title,fontsize=25)
        plt.xlabel(xlabel,fontsize=18)
        plt.ylabel("qubit expected population",fontsize=18)
        plt.ylim((0,1))

    def post_process(self,circuit,i,readout_type='read qubit'):
        '''
        simulate the circuit and get the data
        refresh the plot
        '''
        final_state=self.run_circuit(circuit)
        self.final_state_list.append(final_state)
        if readout_type=='read qubit':
            expected_population=expect(num(self.processor.dims[0]),final_state.ptrace(0))
        elif readout_type=='read phonon':
            expected_population=expect(num(self.processor.dims[1]),final_state.ptrace(1))
            plt.ylim((0,np.max(self.y_array)+0.1))
            plt.ylabel("phonon expected num",fontsize=18)
        else:
            raise NameError('readout object select wrong')
        self.y_array[i]=expected_population
        self.line.set_ydata(self.y_array)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def ideal_phonon_fock(self,fock_number):
        '''
        set the initail state as phonon in specific fock state and qubit in g state
        '''
        self.initial_state=basis(self.processor.dims, [0]+[fock_number]*(self.processor.N-1))

    def ideal_coherent_state(self,alpha):
        '''
        set the initail state as phonon in specific coherent state and qubit in g state
        '''
        self.initial_state=tensor(fock(self.processor.dims[0],0),coherent(self.processor.dims[1],alpha))


    def generate_fock_state(self,fock_number):
        '''
        simulation of using qubit phonon swap to generate phonon fock state
        '''
        self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        for swap_t in self.swap_time_list[:fock_number]:
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=swap_t)
            self.initial_state=self.run_circuit(circuit)
        print('fidelity of phonon : ',expect(self.initial_state.ptrace(1),fock(self.processor.dims[1],fock_number)))
    
    def generate_coherent_state(self,phonon_drive_params=None):
        '''
        simulation of driving phonon mediated by qubit, expect a coherent state
        '''
        if not(phonon_drive_params==None):
            self.phonon_drive_params=phonon_drive_params
        self.initial_state=basis(self.processor.dims, [0]+[0]*(self.processor.N-1))
        circuit = QubitCircuit((self.processor.N))
        circuit.add_gate("XY_R_GB", targets=0,arg_value=self.phonon_drive_params)
        self.initial_state=self.run_circuit(circuit)
    
    

    def phonon_T1_measurement(self):
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='phonon T1')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('swap',targets=[0,1])
            circuit.add_gate('Wait',targets=0,arg_value=t)
            circuit.add_gate('swap',targets=[0,1])
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_T1())


    def phonon_rabi_measurement(self):
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='phonon rabi')
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0)
            circuit.add_gate('Z_R_GB',targets=[0,1],arg_value=t)
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_phonon_rabi())

    def qubit_ramsey_measurement(self,artificial_detuning=None):
        self.x_array=self.t_list
        self.set_up_1D_experiment(title='qubit Ramsey')
        if not(artificial_detuning==None):
            self.artificial_detuning=artificial_detuning
        i=0
        for t in tqdm(self.x_array):
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2})
            circuit.add_gate('Wait',targets=0,arg_value=t)
            circuit.add_gate("X_R", targets=0,arg_value={'rotate_phase':np.pi/2,\
                'rotate_direction':2*np.pi*self.artificial_detuning*t})
            self.post_process(circuit,i)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)
        self.fit_result.append(self.fitter.fit_T2())

    def spec_measurement(self,qubit_probe_params={},readout_type='read qubit'):
        if not(qubit_probe_params=={}):
            self.qubit_probe_params=qubit_probe_params
        self.x_array=self.detuning_list
        self.set_up_1D_experiment(title='qubit spec',xlabel='detuning (MHz)')
        i=0
        for detuning in tqdm(self.x_array):
            self.qubit_probe_params['detuning']=detuning
            circuit = QubitCircuit((self.processor.N))
            circuit.add_gate("XY_R_GB", targets=0,arg_value=self.qubit_probe_params)
            self.post_process(circuit,i,readout_type)
            i=i+1
        self.fitter=hbar_fitting.fitter(self.x_array,self.y_array)




