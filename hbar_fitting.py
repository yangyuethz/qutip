import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss1D
from scipy import signal
from scipy.optimize import curve_fit

def Lorentz(x,x0,w,A,B):
    return A*(1-1/(1+((x-x0)/w)**2))+B

def Cos(x, a, f, os):
    return os + a * np.cos(2 * np.pi * (f * x ))

def Exp_decay(x, A, tau, ofs):
    return A * np.exp(-x / tau) + ofs

def Exp_sine(x, a, tau, ofs, freq , phase):
    return ofs + a * (np.exp(-x/ tau) * np.cos(2 * np.pi * (freq * x + phase)))

def Exp_plus_sine(x, a0, a1,tau1,tau2, ofs, freq , phase):
#     print(a, tau, ofs, freq, phase)
    return ofs + a0 * np.exp(-x/ tau1)*(np.cos(2 * np.pi * (freq * x + phase)))\
        +a1*np.exp(-x/ tau2)

def continue_fourier_transform(time_array,amp_array,freq_array):
    ft_list=[]
    dt=time_array[1]-time_array[0]
    num=len(time_array)
    for w in freq_array:
        phase=np.power((np.zeros(num)+np.exp(-2*np.pi*1j*w*dt)),np.array(range(num)))      
        ft_data=np.sum(amp_array*phase)
        ft_list.append(np.abs(ft_data))
    return np.array(ft_list)


class fitter(object):
    def __init__(self,x_array,y_array):
        self.x_array=x_array
        self.y_array=y_array
    def fit_T1(self):
        #fit for exp decay, e.g. T1
        x_array=self.x_array
        y_array=self.y_array
        minimum_amp=np.min(y_array)
        normalization=y_array[-1]
        popt,pcov =curve_fit(Exp_decay,x_array,y_array,[-(normalization-minimum_amp),20,normalization])
        fig,ax=plt.subplots()
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_decay(x_array,*popt),label='fitted')
        plt.title(('T1 = %.3f us '% (popt[1])))
        plt.legend()
        plt.show()
        return {'T1':popt[1]}
    
    def fit_phonon_rabi(self):
        #fit for phonon_qubit oscillation
        x_array=self.x_array
        y_array=self.y_array
        minimum_point=signal.argrelextrema(y_array, np.less)[0]
        delay_range=x_array[-1]-x_array[0]
        minimum_amp=np.min(y_array)
        max_amp=np.max(y_array)
        freq_guess=1/(x_array[minimum_point[1]]-x_array[minimum_point[0]])

        popt,pcov =curve_fit(Exp_plus_sine,x_array,y_array,[-(max_amp-minimum_amp),0.5,
                                                            delay_range/3,delay_range/3,0,freq_guess,0])
     
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_plus_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title('swap time = %.3f us '% (1/popt[-2]/2))
        plt.show()
        return {'swap_time':1/popt[-2]/2}

    def fit_T2(self):
        x_array=self.x_array
        y_array=self.y_array
        y_smooth=signal.savgol_filter(y_array,51,4)
        y_smooth=signal.savgol_filter(y_smooth,51,4)

        minimum_number=len(*signal.argrelextrema(y_smooth, np.less))
        amp_range=x_array[-1]-x_array[0]
        minimum_amp=np.min(y_array)
        normalization=np.average(y_array)
        popt,pcov =curve_fit(Exp_sine,x_array,y_array,[(normalization-minimum_amp),amp_range/3,normalization,minimum_number/amp_range,0])
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title('T2 = %.3f us, detuning is %.3f MHz '% (popt[1],popt[3]))
        plt.show()
        return {'T2': popt[1],
            'delta': popt[3]
            }
    
    def fit_single_peak(self):
        x_array=self.x_array
        y_array=self.y_array
        max_point=x_array[np.argsort(y_array)[-1]]
        popt,pcov =curve_fit(Lorentz,x_array,y_array,[max_point,1,1,0])
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title(('w0 = %.5f MHz '% (popt[0]/1e9)))
        plt.show()
        return  {'w0': popt[0],
        'width':popt[1],
        }
    

