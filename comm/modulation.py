# =============================================================================
# ========================== (c) Ronald Nissel ================================
# ======================== First version: 29.03.2018 ==========================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

class _MultiCarrierModulation:
    def __init__(self, 
                 nr_subcarriers, 
                 nr_symbols_in_time, 
                 desired_subcarrier_spacing, 
                 sampling_rate,
                 desired_frequency_shift): 

        # Match subcarrier spacing to the sampling rate
        FFTsize = int(np.round(sampling_rate/desired_subcarrier_spacing))
        subcarrier_spacing = sampling_rate/FFTsize       
        if (sampling_rate/desired_subcarrier_spacing % 1) != 0:
            print('Warning, sampling rate divided by subcarrier spacing must be an integer. Thus, the subcarrier spacing is set to', subcarrier_spacing/1e3, 'kHz')       

        # Match frequency shift to the sampling rate        
        circshiftFFT = int(round(desired_frequency_shift/subcarrier_spacing))
        if abs(desired_frequency_shift - circshiftFFT * subcarrier_spacing) > np.finfo(float).eps*10:        
            print('Warning, frequency shift must be a multiple of the subcarrier spacing. Thus, the frequency shift is set to', circshiftFFT*subcarrier_spacing/1e3, 'kHz')
        
        # Integer parameters
        self.nr_subcarriers = nr_subcarriers
        self.nr_symbols_in_time = nr_symbols_in_time
        
        # Parameters with a physical unit
        self.phy_sampling_rate = sampling_rate
        self.phy_dt = 1/sampling_rate        
        self.phy_subcarrier_spacing = subcarrier_spacing
        self.phy_intermediate_frequency = (nr_subcarriers/2 + circshiftFFT) * subcarrier_spacing
        
        # Parameters needed for the implementation
        self.imp_circshiftFFT = circshiftFFT
        self.imp_FFTsize = FFTsize
        self.imp_normalization = np.sqrt(FFTsize / nr_subcarriers)
        
     
class OFDM:
    """ 
    OFDM(L, K, F, fs, foff, Tcp) 
    
    OFDM modulation and demodulation, including SC-FDMA
    
    Parameters
    ----------
    L : Number of subcarriers  
     
    K : Number of OFDM symbols in time
    
    F : Desired subcarrier spacing in Hz. Note that fs/F should be an integer.
    
    fs: Sampling frequency in Hz
    
    foff: Frequency shift in Hz. Should be a multiple of F.
    
    Tcp: Cyclic prefix length in seconds.
   
    Usage
    ----------        
    - Modulate symbols of size (L,K) with ".modulate( symbols )", delivering the OFDM signal in the time domain
    
    - Demodulate the time domain signal with ".demodulate( signal )"
    
    - For SC-FDMA, perform precoding and decoding by ".DFT_precode( symbols )" and ".DFT_decode( symbols )"
    
    """
    def __init__(self, 
                 nr_subcarriers, 
                 nr_symbols_in_time, 
                 desired_subcarrier_spacing, 
                 sampling_rate,
                 desired_frequency_shift,
                 desired_CP_length):
        _MultiCarrierModulation.__init__(self, nr_subcarriers, nr_symbols_in_time, desired_subcarrier_spacing, sampling_rate, desired_frequency_shift)
        
        # Match CP length to the sampling rate
        imp_CP_length = int(np.round(desired_CP_length * self.phy_sampling_rate)) 
        if (desired_CP_length * self.phy_sampling_rate % 1) > np.finfo(float).eps*1e6:
              print('Warning, cylic prefix length times sampling rate must be an integer. Thus, the cylic prefix length is set to', imp_CP_length*self.phy_dt/1e-6, '\u03bcs')       
      
        self.imp_CP_length = imp_CP_length
        
        self.phy_CP_length = imp_CP_length * self.phy_dt
        self.phy_time_spacing = self.phy_CP_length + 1/self.phy_subcarrier_spacing
        
        self.nr_samples =  self.nr_symbols_in_time * (self.imp_FFTsize + self.imp_CP_length)

    def modulate(self, symbols):
        """
        Perform OFDM modulation
    
        Input 
        ----------
        symbols : Array of size (L,K), with L = number of subcarriers and K = number of OFDM symbols in time
    
        Returns
        -------
        out : Array of size (N,1), representing the OFDM signal in the time domain (N = self.nr_samples)
        
        """        
        symbols = symbols.reshape(self.nr_subcarriers, self.nr_symbols_in_time, order='F')
        
        # Determine the IFFT input for OFDM
        ifft_input = np.vstack((symbols, np.zeros((self.imp_FFTsize - self.nr_subcarriers, self.nr_symbols_in_time))))
        ifft_input = np.roll(ifft_input, self.imp_circshiftFFT, axis=0)
        
        # Transmit signal in the time domain
        signal_noCP = np.fft.ifft(ifft_input, axis=0, norm="ortho") * self.imp_normalization
        if self.imp_CP_length>0:
            signal = np.vstack((signal_noCP[-self.imp_CP_length::,:], signal_noCP))
        else:
            signal = signal_noCP
        return signal.reshape(self.nr_samples, 1, order='F')
    
    def demodulate(self, signal):
        """
        Demodulate an OFDM signal
    
        Input
        ----------
        signal : Array of size (N,1), with N = self.nr_samples
    
        Returns
        -------
        out : Array of size (L,K), with L = number of subcarriers and K = number of OFDM symbols in time
        
        """               
        signal = signal.reshape(self.imp_FFTsize + self.imp_CP_length, self.nr_symbols_in_time, order='F')
        
        fft_output = np.fft.fft(signal[-self.imp_FFTsize::, :], axis=0, norm="ortho")        
        fft_output = np.roll(fft_output, -self.imp_circshiftFFT, axis=0)/self.imp_normalization
        
        return fft_output[:self.nr_subcarriers,:]
    
    def DFT_precode(self, symbols):
        """
        DFT precoding => Transforms OFDM into an SCFDMA
    
        Input
        ----------
        symbols : Array of size (L,K), with L = number of subcarriers and K = number of OFDM symbols in time
    
        Returns
        -------
        out : Array of size (L,K)
        
        """         
        symbols = symbols.reshape(self.nr_subcarriers, self.nr_symbols_in_time, order='F')        
        return np.fft.fft(symbols, axis=0, norm="ortho")
    
    def DFT_decode(self, symbols):
        """
        DFT precoding => Transforms OFDM into an SCFDMA
    
        Input
        ----------
        symbols : Array of size (L,K), with L = number of subcarriers and K = number of OFDM symbols in time
    
        Returns
        -------
        out : Array of size (L,K)
        
        """            
        symbols = symbols.reshape(self.nr_subcarriers, self.nr_symbols_in_time, order='F')   
        return np.fft.ifft(symbols, axis=0, norm="ortho")
         

class FBMC:
    """ 
    FBMC(L, K, F, fs, foff, p, o) 
    
    FBMC modulation and demodulation, including pruned DFT spread FBMC
    
    Parameters
    ----------
    L : Number of subcarriers  
     
    K : Number of OFDM symbols in time
    
    F : Subcarrier spacing in Hz. Nfft = fs/F should be an integer.
    
    fs : Sampling frequency in Hz
    
    foff : Frequency shift in Hz. Must be a multiple of F.
    
    p : Prototype filter: 'PHYDYAS', 'TimeRRC', 'Hermite', 'HermiteTrunc1.56', 'HermiteTrunc1.46'
    
    o : Overlapping factor, must be an integer
   
    Usage
    ----------        
    - Modulate symbols of size (L,K) with ".modulate( symbols )", delivering the FBMC signal in the time domain. Note that conventional FBMC, only real-valued symbols can be transmitted.
    
    - Demodulate the time domain signal with ".demodulate( signal )"

    Pruned DFT spread FBMC
    ----------       
    - The prototype filter should be 'HermiteTrunc1.56',  'HermiteTrunc1.46' or 'TimeRRC'
    
    - Initialize with ".prunedDFT_initialize(Lcp)", where Lcp is an even integer and represents the length of the frequency CP. Can usually be set to zero
    
    - Perform precoding and decoding by ".prunedDFT_precoding( symbols )" and ".prunedDFT_decoding( symbols )", similar as in SC-FDMA
    
    """
    def __init__(self, 
                 nr_subcarriers, 
                 nr_symbols_in_time, 
                 desired_subcarrier_spacing, 
                 sampling_rate,
                 desired_frequency_shift,
                 prototype_filter,
                 overlapping_factor):
        _MultiCarrierModulation.__init__(self, nr_subcarriers, nr_symbols_in_time, desired_subcarrier_spacing, sampling_rate, desired_frequency_shift)
  
        # Determine protoypte filter
        t_minmax = overlapping_factor * 1/(2 * self.phy_subcarrier_spacing)
        t = np.arange(-t_minmax, t_minmax, self.phy_dt)
        if prototype_filter ==  "Hermite":  # Orthogonal for TF=2
            prototype_filter_sampled = self._hermite_filter(t, 1/self.phy_subcarrier_spacing, overlapping_factor)
        elif prototype_filter == "PHYDYAS":  # Orthogonal for TF=2
            prototype_filter_sampled = self._PYDYAS_filter(t, 1/self.phy_subcarrier_spacing, overlapping_factor)
        elif prototype_filter == "TimeRRC":  # Orthogonal for TF=2
            prototype_filter_sampled = self._timeRRC_filter(t, 1/self.phy_subcarrier_spacing)
        elif prototype_filter == "HermiteTrunc1.56":  # NOT orthogonal for TF=2
            prototype_filter_sampled = self._hermite_filter(t, 1/self.phy_subcarrier_spacing, 1.56)       
        elif prototype_filter == "HermiteTrunc1.46":  # NOT orthogonal for TF=2
            prototype_filter_sampled = self._hermite_filter(t, 1/self.phy_subcarrier_spacing, 1.46)               
        else:
            print("Please choose a valid protoype filter: \"Hermite\", \"PHYDYAS\", or \"TimeRRC\"" )


        # Precalculate pi/2  phase shift
        k,l = np.meshgrid(np.arange(nr_symbols_in_time),np.arange(nr_subcarriers))
        phase_shift = np.exp(1j * np.pi/2 * (l+k))
        
        # Mapping for the overlap and add operation
        mapping_time = np.zeros((self.imp_FFTsize * overlapping_factor, nr_symbols_in_time), dtype=int)
        for i in range(nr_symbols_in_time):
            mapping_time[:,i] = np.arange(self.imp_FFTsize * overlapping_factor) + i * (self.imp_FFTsize/2)

        
        # Attributes
        self.phy_time_spacing = 1/self.phy_subcarrier_spacing/2
        
        self.imp_prototype_filter_sampled = prototype_filter_sampled.reshape( np.size(prototype_filter_sampled), 1)       
        self.imp_overlapping_factor = overlapping_factor
        self.imp_phase_shift = phase_shift
        self.imp_mapping_time = mapping_time
        self.imp_pDFTsFBMC_onetapscaling = np.array([np.nan])
        self.nr_samples =  int(self.imp_overlapping_factor * self.imp_FFTsize + self.imp_FFTsize/2 * (self.nr_symbols_in_time  - 1))


    def modulate(self, symbols):
        """
        Perform FBMC modulation
    
        Input 
        ----------
        symbols : Array of size (L,K), with L = number of subcarriers and K = number of FBMC symbols in time. Note that conventional FBMC, only real-valued symbols can be transmitted.
    
        Returns
        -------
        out : Array of size (N,1), representing the FBMC signal in the time domain (N = self.nr_samples)
        
        """           
        symbols = symbols.reshape(self.nr_subcarriers, self.nr_symbols_in_time, order='F')
        
        # Phase shift to guarantee real orthogonality
        symbols = symbols * self.imp_phase_shift
        
        # Determine the IFFT input (same as for OFDM)
        ifft_input = np.vstack( (symbols, np.zeros((self.imp_FFTsize - self.nr_subcarriers, self.nr_symbols_in_time))) )
        ifft_input = np.roll(ifft_input, self.imp_circshiftFFT, axis=0)
        
        # IFFT output
        signal_temp = np.fft.ifft(ifft_input, axis=0, norm="ortho") * self.imp_normalization       
        
        # Repeat O times
        signal_rep = np.tile(signal_temp, (self.imp_overlapping_factor, 1))
        
        # Element-wise multiplication with the protoype filter
        signal_rep = signal_rep * self.imp_prototype_filter_sampled

        # Overlap and add in the time domain
        signal_sep = 1j*np.zeros( (self.nr_samples, self.nr_symbols_in_time))
        for i_k in range(0,self.nr_symbols_in_time):              
            signal_sep[self.imp_mapping_time[:,i_k], i_k] = signal_rep[:, i_k]
        signal = np.sum(signal_sep, axis = 1)
        return signal.reshape(self.nr_samples, 1, order='F') 

       
    def demodulate(self, signal): 
        """
        Demodulate an FBMC signal
    
        Input
        ----------
        signal : Array of size (N,1), with N = self.nr_samples
    
        Returns
        -------
        out : Array of size (L,K), with L = number of subcarriers and K = number of FBMC symbols in time
        
        """              
        # Element-wise multiplication by the protoype filter
        signal_temp = signal[self.imp_mapping_time].reshape(self.imp_overlapping_factor*self.imp_FFTsize,self.nr_symbols_in_time, order='F')
        signal_temp = signal_temp * self.imp_prototype_filter_sampled
        
        # Summing up sample (reverse operation of O time repetition)
        signal_temp2 = np.sum(signal_temp.reshape(self.imp_FFTsize, self.imp_overlapping_factor, self.nr_symbols_in_time, order='F'), axis=1)   
        
        # FFT 
        symbols_temp = np.fft.fft(signal_temp2, axis=0, norm="ortho")/self.imp_normalization
        
        # Frequency offset compensation and taking the correct elements 
        symbols_temp = np.roll(symbols_temp, -self.imp_circshiftFFT, axis=0)
        symbols = symbols_temp[:self.nr_subcarriers,:] * np.conj(self.imp_phase_shift)
        return symbols
 
    
    def oqam_staggering(self, symbols_qam):
        """
        OQAM staggering: map the real part of a complex-valued symbol to the first time-slot and the imaginary part to the second time-slot. This is required because FBMC only allows the transmission of real-valued symbols.
    
        Input
        ----------
        symbols_qam : Complex-valued array of size (L,K/2)
    
        Returns
        -------
        out : Real-valued array of size (L,K)
        
        """           
        symbols_qam = symbols_qam.reshape(self.nr_subcarriers, int(self.nr_symbols_in_time/2), order='F')
        symbols = np.empty( (self.nr_subcarriers, self.nr_symbols_in_time) ) 
        symbols[:,0::2] = np.real(symbols_qam)
        symbols[:,1::2] = np.imag(symbols_qam)
        return symbols

    
    def oqam_destaggering(self, symbols):
        """
        OQAM destaggering: the first time-slot is mapped to the real part of a complex-valued symbol and the second time-slot to the imaginary part. 
    
        Input
        ----------
        symbols : Real-valued array of size (L,K)
    
        Returns
        -------
        out : Complex-valued array of size (L,K/2)
        
        """           
        symbols_qam =  np.real(symbols[:,0::2]) + 1j*np.real(symbols[:,1::2])
        return symbols_qam  
   
    
    def prunedDFT_initialize(self, nr_CP_subcarriers):
        """
        Initialize pruned DFT spread FBMC, that is, find the one-tap scaling values.
    
        Parameters
        ----------
        nr_CP_subcarriers : Number of frequency subcarriers. Can usually be set to zero.
        
        """          
        self.imp_pDFTsFBMC_nr_CP_subcarriers = nr_CP_subcarriers
        self.imp_pDFTsFBMC_datasub = int((self.nr_subcarriers - nr_CP_subcarriers)/2)
        self.imp_pDFTsFBMC_onetapscaling = np.ones((self.imp_pDFTsFBMC_datasub,1))
        
        # Calculate one-tap coefficients
        onetapscaling = np.zeros((self.imp_pDFTsFBMC_datasub,1))
        for i in range(self.imp_pDFTsFBMC_datasub):
            x = np.zeros( (self.imp_pDFTsFBMC_datasub, self.nr_symbols_in_time) )
            x[i,0] = 1
            y = self.prunedDFT_decoding(self.demodulate(self.modulate(self.prunedDFT_precoding(x))))  
            onetapscaling[i] = 1/abs(y[i,0])
            
        self.imp_pDFTsFBMC_onetapscaling = np.sqrt(onetapscaling)   
        
 
    def prunedDFT_precoding(self, symbols):
        """
        Apply one-tap scaling and pruned DFT precoding => transform FBMC into pruned DFT spread FBMC, similar as SC-FDMA in OFDM
        
        Input
        ----------
        symbols : Array of size ((L-Lcp)/2,K)
    
        Returns
        -------
        out : Array of size (L,K)
        
        """
        if np.isnan(self.imp_pDFTsFBMC_onetapscaling[0]):
            self.prunedDFT_initialize(0)
            
        symbols = symbols.reshape(self.imp_pDFTsFBMC_datasub, self.nr_symbols_in_time, order='F')
        
        # One-tap scaling
        symbols = symbols * self.imp_pDFTsFBMC_onetapscaling
        
        # Pruned DFT spread (only half the input samples are data, the other half is set to zero)
        if self.imp_overlapping_factor % 2:
            fft_input = np.vstack( (np.zeros( (self.imp_pDFTsFBMC_datasub, self.nr_symbols_in_time) ), symbols) )
        else:
            fft_input = np.vstack( (symbols, np.zeros( (self.imp_pDFTsFBMC_datasub, self.nr_symbols_in_time) )) )
        fft_output = np.fft.fft(fft_input, axis=0, norm="ortho") 
              
        # Add frequency cyclic prefix
        if self.imp_pDFTsFBMC_nr_CP_subcarriers>0:
            return np.vstack( (fft_output[-int(self.imp_pDFTsFBMC_nr_CP_subcarriers/2):,:],
                           fft_output,
                           fft_output[0:int(self.imp_pDFTsFBMC_nr_CP_subcarriers/2),:]) ) 
        else:
            return fft_output
         
        
    def prunedDFT_decoding(self, symbols):
        """
        Inverse operation of pruned DFT precoding
        
        Input
        ----------
        symbols : Array of size (L,K)
    
        Returns
        -------
        out : Array of size ((L-Lcp)/2,K)
        
        """        
        if np.isnan(self.imp_pDFTsFBMC_onetapscaling[0]):
            self.prunedDFT_initialize(0)
            
        symbols = symbols.reshape(self.nr_subcarriers, self.nr_symbols_in_time, order='F')
        
        # Remove frequency cyclic prefix
        if self.imp_pDFTsFBMC_nr_CP_subcarriers > 0:
            symbols_noCP = symbols[int(self.imp_pDFTsFBMC_nr_CP_subcarriers/2):-int(self.imp_pDFTsFBMC_nr_CP_subcarriers/2),:]
        else:
            symbols_noCP = symbols
        
        # Inverse pruned DFT spread
        ifft_output = np.fft.ifft(symbols_noCP, axis=0, norm="ortho")
        if self.imp_overlapping_factor % 2:
            despreaded_symbols = ifft_output[-self.imp_pDFTsFBMC_datasub:,:]
        else:
            despreaded_symbols = ifft_output[:self.imp_pDFTsFBMC_datasub,:] 
        
        # One-tap scaling
        return despreaded_symbols * self.imp_pDFTsFBMC_onetapscaling
      
    def _timeRRC_filter(self, t, T0):    
        return np.logical_and(t<=(T0/2), t>-T0/2) * np.sqrt( 1 + ( np.cos(np.pi*1*2*t/T0) ) ) # / np.sqrt(T0);   

    def _PYDYAS_filter(self, t, T0, O):    
        if O==3:
            print("To do")
        elif O==4:
            return np.logical_and(t<=(4*T0/2), t>-4*T0/2) * (1+2*(
                    0.97195983 *    np.cos(2*np.pi*1/4*t/T0) +
                    np.sqrt(2)/2  * np.cos(2*np.pi*2/4*t/T0) + 
                    0.23514695 *    np.cos(2*np.pi*3/4*t/T0) ))/np.sqrt(4**2)
        elif O==8:
            print("To do")
        else:
            print("Error, overlapping factor must be 4") 

    def _hermite_filter(self, t, T0, O):
         return np.logical_and(t<=(O*T0/2), t>-O*T0/2) * np.exp(-np.pi * (t/(T0/np.sqrt(2)))**2) * (
                1.412692577 + 
                -3.0145e-3 * ((12+(-48) * (np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**2+16*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**4 ) ) + 
                -8.8041e-6 * (1680+(-13440)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**2+13440*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**4+(-3584)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**6+256*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**8 ) + 
                -2.2611e-9 * (665280+(-7983360)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**2+13305600*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**4+(-7096320)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**6+1520640* (np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**8+(-135168)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**10+4096*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**12 ) + 
                -4.4570e-15 * (518918400+(-8302694400)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**2+19372953600*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**4+(-15498362880) * (np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**6+5535129600*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**8+(-984023040)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**10+89456640*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**12+(-3932160)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**14+65536*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**16 ) + 
                1.8633e-16 * (670442572800+(-13408851456000)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**2+40226554368000*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**4+(-42908324659200)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**6+21454162329600*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**8+(-5721109954560)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**10+866834841600*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**12+(-76205260800)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**14+3810263040*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**16+ (-99614720)*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**18+1048576*(np.sqrt(2*np.pi)*(t/(T0/np.sqrt(2))))**20 ))
                

    
    
    
    
    
    
    