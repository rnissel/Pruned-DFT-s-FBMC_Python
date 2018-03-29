# =============================================================================
# ========================== (c) Ronald Nissel ================================
# ======================== First version: 29.03.2018 ==========================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


class DoublySelectiveChannel:
    """ 
    DoublySelectiveChannel(pdp, nu_max, fs, N, nrp) 
    
    OFDM modulation and demodulation, including SC-FDMA
    
    Parameters
    ----------
    pdp : Power delay profile, see below.
     
    nu_max : Maximum doppler shift in Hz.
        
    fs: Sampling rate (Hz). Should approximately match the predefined power delay profile.
    
    N: Maximum number of samples in the time domain.
    
    nrp: Number of multipath propagations for the WSSUS process
   
    Usage
    ----------        
    - ".new_realization()" generates a new random channel realizatio
    
    - ".convolution( signal )" performs a convolution of the signal with the time-variant impulse response of the channel.
    
    Power delay profile
    ----------       
    - 'TDL-A_xxns', with 'xx' beeing the rms delay spread in ns.
    - 'TDL-B_xxns', with 'xx' beeing the rms delay spread in ns.
    - 'TDL-C_xxns', with 'xx' beeing the rms delay spread in ns.
    - 'VehicularA'
    - 'PedestrianA'
    - 'Flat', doubly-flat Rayleigh fading (not a doubly selective channel)
    - 'AWGN', pure AWGN channel (not a doubly selective channel)
      
    """
    def __init__( self, 
                  power_delay_profile,
                  max_doppler_shift,
                  sampling_rate,
                  nr_samples,
                  nr_paths_wssus):
   
        # Determine the power delay profile
        if power_delay_profile[:3]=='TDL':
            pos1 = power_delay_profile.find('_')
            pos2 = power_delay_profile.find('ns') 
            desired_rms_delay_spread = int(power_delay_profile[pos1+1:pos2])*10**(-9)
            if power_delay_profile[4] == 'A':
                ref_pdp_dB = np.array([-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,-12.4,-15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7])
                ref_pdp_delay = desired_rms_delay_spread * np.array([0.0000,0.3819,0.4025,0.5868,0.4610,0.5375,0.6708,0.5750,0.7618,1.5375,1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,4.0810,4.4579,4.5695,4.7966,5.0066,5.3043,9.6586])
            elif power_delay_profile[4] == 'B':
                ref_pdp_dB = np.array([0,-2.2,-4,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3,-8.9,-9,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,-11.4,-14.9,-9.2,-11.3])
                ref_pdp_delay = desired_rms_delay_spread * np.array([0.0000,0.1072,0.2155,0.2095,0.2870,0.2986,0.3752,0.5055,0.3681,0.3697,0.5700,0.5283,1.1021,1.2756,1.5474,1.7842,2.0169,2.8294,3.0219,3.6187,4.1067,4.2790,4.7834])
            elif power_delay_profile[4] == 'C':
                ref_pdp_dB = np.array([-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16,-15.7,-21.6,-22.8])
                ref_pdp_delay = desired_rms_delay_spread * np.array([0,0.2099,0.2219,0.2329,0.2176,0.6366,0.6448,0.6560,0.6584,0.7935,0.8213,0.9336,1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523])   
        else:
            if power_delay_profile == 'VehicularA':
                ref_pdp_dB = np.array([0,-1,-9,-10,-15,-20])
                ref_pdp_delay = np.array([0,310e-9,710e-9,1090e-9,1730e-9,2510e-9])
            elif power_delay_profile == 'PedestrianA':
                ref_pdp_dB = np.array([0,-9.7,-19.2,-22.8])
                ref_pdp_delay = np.array([0,110e-9,190e-9,410e-9])
            elif power_delay_profile == 'Flat':
                ref_pdp_dB = np.array([0])
                ref_pdp_delay = np.array([0])
            elif power_delay_profile == 'AWGN':
                ref_pdp_dB = np.array([0])
                ref_pdp_delay = np.array([0])                
            else:
                print("Channel model \"" + power_delay_profile + "\" is not supported")
               
        # Fit the channel delay taps to the sampling rate and normalize the power delay profile    
        index_pdp_temp = np.round(ref_pdp_delay*sampling_rate).astype(int)
        pdp_temp = np.zeros((max(index_pdp_temp)+1,index_pdp_temp.size))
        for i_index in range(index_pdp_temp.size):
            pdp_temp[index_pdp_temp[i_index],i_index] = 10**(ref_pdp_dB[i_index]/10) 
        pdp = np.sum(pdp_temp,axis=1)/np.sum(pdp_temp)
        pdp_index = np.unique(np.sort(index_pdp_temp))        
    
        # Calculate RMS delay spread 
        dt = 1/sampling_rate
        tau = np.arange(0,len(pdp)) * dt
        mean_delay = sum(tau*pdp)
        rms_delay_spread = np.sqrt(sum(tau**2 * pdp) - mean_delay**2);       
        if (ref_pdp_delay/dt % 1 > np.finfo(float).eps *10).any():
            print('Sampling rate does not fit the channel model => RMS delay spread is changed from', str(round(desired_rms_delay_spread * 10**9)) +'ns to', str(int(round(rms_delay_spread * 10**9))) + 'ns')
    
        # Atributes
        self.imp_pdp = pdp
        self.imp_pdp_index = pdp_index
        self.imp_pdp_string = power_delay_profile        
        self.imp_nr_paths_wssus = nr_paths_wssus        
        self.phy_max_doppler_shift = max_doppler_shift
        self.phy_sampling_rate = sampling_rate
        self.phy_dt = dt        
        self.nr_samples = nr_samples    
        
        # Generat random channel
        self.new_realization()
    
    def new_realization(self):
        """
        Generate a new channel realization and save the time-variant convolution matrix in "self.imp_convolution_matrix" 
        
        """       
        if self.imp_pdp_string=='AWGN':
            impulse_response_squeezed = np.ones((self.nr_samples,1))
        elif self.imp_pdp_string=='Flat':
            h = np.random.randn(1) +  1j * np.random.randn(1)
            impulse_response_squeezed = h * np.ones((self.nr_samples,1))
        else:
            impulse_response_squeezed = np.zeros((self.nr_samples,self.imp_pdp_index.size), dtype=np.complex)
            t = np.arange(self.nr_samples).reshape(self.nr_samples,1)*self.phy_dt
    
            # Use a for loop because it is more memory efficient (but takes longer)
            for i in range(self.imp_nr_paths_wssus):
                doppler_shifts = np.cos(np.random.random((1,self.imp_pdp_index.size))*2*np.pi) * self.phy_max_doppler_shift 
                random_phase = np.random.random((1,self.imp_pdp_index.size))  
                argument = 2 * np.pi * (random_phase + doppler_shifts*t)  
                impulse_response_squeezed += np.cos(argument) + 1j*np.sin(argument)  
            impulse_response_squeezed *= np.sqrt(self.imp_pdp[self.imp_pdp_index]/self.imp_nr_paths_wssus)
        
        self.imp_convolution_matrix = sparse.dia_matrix((np.transpose(impulse_response_squeezed), -self.imp_pdp_index), shape=(self.nr_samples, self.nr_samples)).tocsr()

    def convolution(self, signal):
        """
        Perform convolution of the signal with the channel impulse response, saved in "self.imp_convolution_matrix" 
        
        Input
        ----------
        signal : Array of size (M,1), with M<= N = self.nr_samples
    
        Returns
        -------
        out : Array of size (M,1)
        
        """        
        signal_length = signal.shape[0]
        return self.imp_convolution_matrix[:signal_length,:signal_length].dot(signal)
        
    def get_transfer_function(self, index_time_positions, FFTsize, index_active_subcarriers):   
        """
        Calculates the channel transfer function
        
        Parameters
        ----------
        index_time_positions : array, representing the time index for which the channel transfer function is calculated
        
        FFTsize : integer, fft size of the OFDM and FBMC systems
            
        index_active_subcarriers: arraym, representing the subcarrier index for which the channel transfer function is calculated
    
        Returns
        -------
        out : Array of size (len(index_time_positions), len(index_active_subcarriers))
        
        """           
        transfer_function = np.zeros( (index_time_positions.size, index_active_subcarriers.size) , dtype=complex)
        for i_n in range(index_time_positions.size):
            conv_row = self.imp_convolution_matrix[index_time_positions[i_n],:index_time_positions[i_n]+1]
    
            if conv_row.shape[1]>FFTsize:
                impulse_response = np.fliplr(conv_row[:,-FFTsize:].toarray())
            else:
                impulse_response = np.hstack( (np.fliplr(conv_row.toarray()), np.zeros((1,FFTsize-conv_row.shape[1]))) )
    
            transfer_function[i_n,:] = np.fft.fft(impulse_response)[:,index_active_subcarriers]

        return transfer_function.transpose()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        