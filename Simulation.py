# =============================================================================
# ========================== (c) Ronald Nissel ================================
# ======================== First version: 29.03.2018 ==========================
# =============================================================================
# This script simulates the BER over SNR for OFDM, SC-FDMA, FBMC and pruned DFT 
# spread FBMC in a doubly selective channel. Furthermore, it simulates the  
# PAPR, the average transmit power over time, and the power spectral density. 
# Note that the simulation parameters are chosen similar to the LTE standard.


import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Self defined classes in the folder "./comm"
from comm.qam import QAM
from comm.modulation import OFDM
from comm.modulation import FBMC
from comm.channel import DoublySelectiveChannel


# =============================================================================
# ====================== Simulation Parameters ================================
# =============================================================================
QAM_order   = 16                            # 4, 16, 64, 256, 1024, ...

F           = 15e3                          # Subcarrier spacing in Hz
L           = 256                           # Number of subcarriers
K_OFDM      = 14                            # Number of OFDM symbols
K_OFDM_noCP = 15                            # Number of OFDM symbols (without CP, to have the same data rate as FBMC!)
K_FBMC      = 30                            # Number of FBMC symbols, must be a multiple of 2

fs          = 15e3*14*96                    # Sampling rate in Hz, should be a multiple of 15e3 and 14 (due to CP). Furthermore, fs/F>=L and fs/F/2 an integer.

T_CP        = 1/F/14                        # Cyclic prefix length in seconds (for OFDM).
L_CP        = 0                             # Number of CP subcarriers (for pruned DFT spread FBMC, can usually be set to zero => no efficiency loss). Must be even. 


pdp         = "TDL-A_300ns"                 # Power delay profile: 'TDL-A_xxns', 'TDL-B_xxns', 'TDL-C_xxns', with 'xx' being the rms delay spread in ns. Further choices 'VehicularA', 'PedestrianA', 'Flat', 'AWGN'
v_kmh       = 200                           # Velocity in km/h
fc          = 2.5e9                         # Carrier frequency in Hz


nr_rep      = 100                          # Number of Monte Carlo repetitions 
SNR_dB      = np.linspace(0,30,21)          # SNR values
save_png    = False                         # If set to true, save Figure 2-5 in "./png/


# =============================================================================
# ==================== Generate Communication Objects =========================
# =============================================================================
# QAM object
qam = QAM(QAM_order)

# OFDM object
ofdm       = OFDM( L,                       # Number of subcarriers
                   K_OFDM,                  # Number of OFDM symbols in time
                   F,                       # Subcarrier spacing (Hz)
                   fs,                      # Sampling rate (Hz), should be multiple of F
                   0,                       # Frequency shift (Hz)
                   T_CP )                   # Cylic prefix length (s), T_CP*fs should be an integer

# OFDM object, without CP
ofdm_noCP  = OFDM( L,                       # Number of subcarriers
                   K_OFDM_noCP,             # Number of OFDM symbols in time
                   F,                       # Subcarrier spacing (Hz)
                   fs,                      # Sampling rate (Hz), should be multiple of F
                   0,                       # Frequency shift (Hz)
                   0 )                      # Cylic prefix length (s)

# FBMC objects
fbmc       = FBMC( L,                       # Number of subcarriers
                   K_FBMC,                  # Number of FBMC symbols in time
                   F,                       # Subcarrier spacing (Hz)
                   fs,                      # Sampling rate (Hz), should be multiple of F
                   0,                       # Frequency shift (Hz)
                   'Hermite',               # Prototype filter: 'PHYDYAS', 'TimeRRC', 'Hermite'
                   4 )                      # Overlapping factor, should be an integer (implementation wise)

# FBMC object, using a truncated prototype filter
fbmc_trunc = FBMC( L,                       # Number of subcarriers
                   K_FBMC,                  # Number of FBMC symbols in time
                   F,                       # Subcarrier spacing (Hz)
                   fs,                      # Sampling rate (Hz), should be multiple of F
                   0,                       # Frequency shift (Hz)
                   'HermiteTrunc1.56',      # 'HermiteTrunc1.56', 'HermiteTrunc1.46'
                   4 )                      # Overlapping factor, should be an integer (implementation wise)

fbmc_trunc.prunedDFT_initialize(L_CP)       # Initialize pruned DFT spread FBMC


# Channel Object
N = max(ofdm.nr_samples, ofdm_noCP.nr_samples, fbmc.nr_samples, fbmc_trunc.nr_samples)
channel = DoublySelectiveChannel(
                   pdp,                     # Power delay profile: 'TDL-A_xxns', 'TDL-B_xxns', 'TDL-C_xxns', with 'xx' beeing the rms delay spread in ns. Further choices 'VehicularA', 'PedestrianA', 'Flat'
                   v_kmh/3.6*fc/2.998e8,    # Maximum doppler shift
                   fs,                      # Sampling rate (Hz). Should approximately match the predefined power delay profile
                   N,                       # Maximum number of samples in the time domain
                   50 )                     # Number of multipath propagations for the WSSUS process


# =============================================================================
# ==================== Preallocate/Precalculate Stuff =========================
# =============================================================================
Pt_OFDM      = np.zeros( (ofdm.nr_samples,1) )
Pt_FBMC      = np.zeros( (fbmc.nr_samples,1) )
Pt_pDFTsFBMC = np.zeros( (fbmc_trunc.nr_samples,1) )

Pf_OFDM      = np.zeros( (ofdm.nr_samples,1) )
Pf_FBMC      = np.zeros( (fbmc.nr_samples,1) )
Pf_pDFTsFBMC = np.zeros( (fbmc_trunc.nr_samples,1) )

PAPR_OFDM      = np.zeros( (nr_rep, K_OFDM) )
PAPR_SCFDMA    = np.zeros( (nr_rep, K_OFDM) )
PAPR_FBMC      = np.zeros( (nr_rep, int(K_FBMC/2)))
PAPR_pDFTsFBMC = np.zeros( (nr_rep, int(K_FBMC/2)))

BER_OFDM       = np.zeros( (SNR_dB.size, nr_rep) )
BER_OFDMnoCP   = np.zeros( (SNR_dB.size, nr_rep) )
BER_SCFDMA     = np.zeros( (SNR_dB.size, nr_rep) )
BER_SCFDMAnoCP = np.zeros( (SNR_dB.size, nr_rep) )
BER_FBMC       = np.zeros( (SNR_dB.size, nr_rep) )
BER_pDFTsFBMC  = np.zeros( (SNR_dB.size, nr_rep) )

time_pos_mid_OFDM     = np.arange(ofdm.nr_symbols_in_time) * (ofdm.imp_CP_length + ofdm.imp_FFTsize) + ofdm.imp_CP_length + int(ofdm.imp_FFTsize/2)
time_pos_mid_OFDMnoCP = np.arange(ofdm_noCP.nr_symbols_in_time) * (ofdm_noCP.imp_CP_length + ofdm_noCP.imp_FFTsize) + ofdm_noCP.imp_CP_length + int(ofdm_noCP.imp_FFTsize/2)
time_pos_mid_FBMC     = np.arange(fbmc.nr_symbols_in_time) * (int(fbmc.imp_FFTsize/2)) + int((fbmc.imp_FFTsize * fbmc.imp_overlapping_factor)/2)


# =============================================================================
# ========================== Start Simulation =================================
# =============================================================================
tic = time.time()
for i_rep in range(nr_rep):
    # Generate new channel realization 
    channel.new_realization()
    
    # Generate random bitstream
    bitstream_OFDM      = np.random.randint(0, 2, ( L * K_OFDM,               qam.bits_per_symbol))    
    bitstream_OFDMnoCP  = np.random.randint(0, 2, ( L * K_OFDM_noCP,          qam.bits_per_symbol))    
    bitstream_FBMC      = np.random.randint(0, 2, ( L * int(K_FBMC/2),        qam.bits_per_symbol))
    bitstream_pDFTsFBMC = np.random.randint(0, 2, ( int((L-L_CP)/2) * K_FBMC, qam.bits_per_symbol))
    
    # Data symbols (map bits to symbols)
    xD_OFDM       = qam.convert_bits2symbols( bitstream_OFDM ).reshape(L, K_OFDM, order='F')
    xD_SCFDMA     = xD_OFDM   
    xD_OFDMnoCP   = qam.convert_bits2symbols( bitstream_OFDMnoCP ).reshape(L, K_OFDM_noCP, order='F')
    xD_SCFDMAnoCP = xD_OFDMnoCP      
    xD_FBMC       = qam.convert_bits2symbols( bitstream_FBMC ).reshape(L, int(K_FBMC/2), order='F')
    xD_pDFTsFBMC  = qam.convert_bits2symbols( bitstream_pDFTsFBMC ).reshape(int((L-L_CP)/2), K_FBMC, order='F') 
    
    # Transmitted symbols (multi-carrier domain)
    x_OFDM       = xD_OFDM
    x_SCFDMA     = ofdm.DFT_precode( xD_SCFDMA ) 
    x_OFDMnoCP   = xD_OFDMnoCP
    x_SCFDMAnoCP = ofdm_noCP.DFT_precode( xD_SCFDMAnoCP )     
    x_FBMC       = fbmc.oqam_staggering( xD_FBMC )
    x_pDFTsFBMC  = fbmc_trunc.prunedDFT_precoding( xD_pDFTsFBMC )
    
    # Transmitted signal in the time domain
    s_OFDM       = ofdm.modulate( x_OFDM )
    s_SCFDMA     = ofdm.modulate( x_SCFDMA )  
    s_OFDMnoCP   = ofdm_noCP.modulate( x_OFDMnoCP )
    s_SCFDMAnoCP = ofdm_noCP.modulate( x_SCFDMAnoCP )      
    s_FBMC       = fbmc.modulate( x_FBMC )
    s_pDFTsFBMC  = fbmc_trunc.modulate( x_pDFTsFBMC )
    
    # Received signal after transmission over a doubly-selective channel (without noise)
    r_OFDM_noNoise       = channel.convolution( s_OFDM )
    r_SCFDMA_noNoise     = channel.convolution( s_SCFDMA )
    r_OFDMnoCP_noNoise   = channel.convolution( s_OFDMnoCP )
    r_SCFDMAnoCP_noNoise = channel.convolution( s_SCFDMAnoCP )    
    r_FBMC_noNoise       = channel.convolution( s_FBMC )
    r_pDFTsFBMC_noNoise  = channel.convolution( s_pDFTsFBMC )
    
    # Find the one-tap channel (time-variant transfer function), required for the equalization later 
    h_OFDM     = channel.get_transfer_function( time_pos_mid_OFDM,     ofdm.imp_FFTsize,      np.arange(L) )
    h_OFDMnoCP = channel.get_transfer_function( time_pos_mid_OFDMnoCP, ofdm_noCP.imp_FFTsize, np.arange(L) )
    h_FBMC     = channel.get_transfer_function( time_pos_mid_FBMC,     fbmc.imp_FFTsize,      np.arange(L) )

    # Pregenerate unit power noise samples 
    noise_OFDM     = 1/np.sqrt(2)* (np.random.randn(s_OFDM.size,1)     +  1j * np.random.randn(s_OFDM.size,1)) 
    noise_OFDMnoCP = 1/np.sqrt(2)* (np.random.randn(s_OFDMnoCP.size,1) +  1j * np.random.randn(s_OFDMnoCP.size,1)) 
    noise_FBMC     = 1/np.sqrt(2)* (np.random.randn(s_FBMC.size,1)     +  1j * np.random.randn(s_FBMC.size,1))     
    
    # Simulate over different noise power levels
    for i_snr in range(SNR_dB.size):
        # Add noise
        Pn =  10**(-SNR_dB[i_snr]/10)   # Symbol noise power
        Pn_time = Pn * fs/F/L           # Noise power in the time domain (Nfft/L)
        r_OFDM       = r_OFDM_noNoise       + np.sqrt(Pn_time) * noise_OFDM
        r_SCFDMA     = r_SCFDMA_noNoise     + np.sqrt(Pn_time) * noise_OFDM
        r_OFDMnoCP   = r_OFDMnoCP_noNoise   + np.sqrt(Pn_time) * noise_OFDMnoCP
        r_SCFDMAnoCP = r_SCFDMAnoCP_noNoise + np.sqrt(Pn_time) * noise_OFDMnoCP        
        r_FBMC       = r_FBMC_noNoise       + np.sqrt(Pn_time) * noise_FBMC
        r_pDFTsFBMC  = r_pDFTsFBMC_noNoise  + np.sqrt(Pn_time) * noise_FBMC

        # Received symbols after demodulation
        y_OFDM       = ofdm.demodulate( r_OFDM )
        y_SCFDMA     = ofdm.demodulate( r_SCFDMA )
        y_OFDMnoCP   = ofdm_noCP.demodulate( r_OFDMnoCP )
        y_SCFDMAnoCP = ofdm_noCP.demodulate( r_SCFDMAnoCP )        
        y_FBMC       = fbmc.demodulate( r_FBMC )
        y_pDFTsFBMC  = fbmc_trunc.demodulate( r_pDFTsFBMC )        
        
        # Calculate the one-tap equalizer
        e_OFDM       = 1/h_OFDM
        e_SCFDMA     = np.conj(h_OFDM) / (np.abs(h_OFDM)**2 + Pn) * 1 / np.mean(1/(1+Pn/np.abs(h_OFDM)**2), axis=0)  
        e_OFDMnoCP   = 1/h_OFDMnoCP
        e_SCFDMAnoCP = np.conj(h_OFDMnoCP) / (np.abs(h_OFDMnoCP)**2 + Pn) * 1 / np.mean(1/(1+Pn/np.abs(h_OFDMnoCP)**2), axis=0)          
        e_FBMC       = 1/h_FBMC
        e_pDFTsFBMC  = np.conj(h_FBMC) / (np.abs(h_FBMC)**2 + Pn) * 1 / np.mean(1/(1+Pn/np.abs(h_FBMC)**2), axis=0)  

        # One-tap equalization (estimation of the transmitted symbols)
        xest_OFDM       = e_OFDM       * y_OFDM
        xest_SCFDMA     = e_SCFDMA     * y_SCFDMA
        xest_OFDMnoCP   = e_OFDMnoCP   * y_OFDMnoCP
        xest_SCFDMAnoCP = e_SCFDMAnoCP * y_SCFDMAnoCP 
        xest_FBMC       = e_FBMC       * y_FBMC
        xest_pDFTsFBMC  = e_pDFTsFBMC  * y_pDFTsFBMC
        
        # Estimated data symbols
        xDest_OFDM       = xest_OFDM
        xDest_SCFDMA     = ofdm.DFT_decode( xest_SCFDMA )
        xDest_OFDMnoCP   = xest_OFDMnoCP
        xDest_SCFDMAnoCP = ofdm_noCP.DFT_decode( xest_SCFDMAnoCP )        
        xDest_FBMC       = fbmc.oqam_destaggering( xest_FBMC )
        xDest_pDFTsFBMC  = fbmc_trunc.prunedDFT_decoding( xest_pDFTsFBMC )
           
        # Quantization of the estimated data symbols and mapping to bits
        detected_bitstream_OFDM       = qam.convert_symbols2bits( xDest_OFDM )
        detected_bitstream_SCFDMA     = qam.convert_symbols2bits( xDest_SCFDMA )
        detected_bitstream_OFDMnoCP   = qam.convert_symbols2bits( xDest_OFDMnoCP )
        detected_bitstream_SCFDMAnoCP = qam.convert_symbols2bits( xDest_SCFDMAnoCP )
        detected_bitstream_FBMC       = qam.convert_symbols2bits( xDest_FBMC )
        detected_bitstream_pDFTsFBMC  = qam.convert_symbols2bits( xDest_pDFTsFBMC )
        
        # Calculate the bit error ratio
        BER_OFDM[i_snr, i_rep]       = np.mean(detected_bitstream_OFDM       != bitstream_OFDM)
        BER_SCFDMA[i_snr, i_rep]     = np.mean(detected_bitstream_SCFDMA     != bitstream_OFDM)
        BER_OFDMnoCP[i_snr, i_rep]   = np.mean(detected_bitstream_OFDMnoCP   != bitstream_OFDMnoCP)
        BER_SCFDMAnoCP[i_snr, i_rep] = np.mean(detected_bitstream_SCFDMAnoCP != bitstream_OFDMnoCP)        
        BER_FBMC[i_snr, i_rep]       = np.mean(detected_bitstream_FBMC       != bitstream_FBMC)
        BER_pDFTsFBMC[i_snr, i_rep]  = np.mean(detected_bitstream_pDFTsFBMC  != bitstream_pDFTsFBMC)    
        
        
    # =========================================================================
    # ============ Additional Calculations for Illustration ===================
    # =========================================================================
    # Calculate the PAPR (average power is always one)
    PAPR_OFDM[i_rep,:]      = np.max(np.abs(s_OFDM.reshape( (ofdm.imp_FFTsize + ofdm.imp_CP_length) , K_OFDM, order='F'))**2, axis=0)    
    PAPR_SCFDMA[i_rep,:]    = np.max(np.abs(s_SCFDMA.reshape( (ofdm.imp_FFTsize + ofdm.imp_CP_length) , K_OFDM, order='F'))**2, axis=0)    
    cutoff = int((fbmc.imp_overlapping_factor-0.5) * fbmc.imp_FFTsize/2)    
    PAPR_FBMC[i_rep,:]      = np.max(np.abs(s_FBMC[cutoff:-cutoff].reshape( (fbmc.imp_FFTsize) , int(K_FBMC/2), order='F'))**2, axis=0)
    PAPR_pDFTsFBMC[i_rep,:] = np.max(np.abs(s_pDFTsFBMC[cutoff:-cutoff].reshape( (fbmc.imp_FFTsize), int(K_FBMC/2), order='F'))**2, axis=0)

    # Transmit power in time
    Pt_OFDM      = Pt_OFDM      + np.abs(s_OFDM)**2
    Pt_FBMC      = Pt_FBMC      + np.abs(s_FBMC)**2
    Pt_pDFTsFBMC = Pt_pDFTsFBMC + np.abs(s_pDFTsFBMC)**2
      
    # Transmit power in frequency (power spectral density)
    Pf_OFDM      = Pf_OFDM      + np.abs(np.fft.fft(s_OFDM, axis=0))**2
    Pf_FBMC      = Pf_FBMC      + np.abs(np.fft.fft(s_FBMC, axis=0))**2
    Pf_pDFTsFBMC = Pf_pDFTsFBMC + np.abs(np.fft.fft(s_pDFTsFBMC, axis=0))**2
    
    # Status indicator
    toc = time.time() - tic
    if i_rep == 0 or ((i_rep +1) % 10) == 0:
        print("%3d" %  ( int((i_rep + 1)/nr_rep * 100) ) + '% ... ' + str(round(toc/(i_rep+1)*(nr_rep-i_rep-1)/60,1)) + 'minutes left')

        

# =============================================================================
# ========================== Post-processing ==================================
# =============================================================================   
# Calculate the average power in time
Pt_OFDM      = Pt_OFDM / nr_rep  
Pt_FBMC      = Pt_FBMC / nr_rep  
Pt_pDFTsFBMC = Pt_pDFTsFBMC / nr_rep  

# Normalize power in frequency to 0dB (simpleton methon)
Pf_OFDM      = Pf_OFDM / max(Pf_OFDM)    
Pf_FBMC      = Pf_FBMC / max(Pf_FBMC)
Pf_pDFTsFBMC = Pf_pDFTsFBMC / max(Pf_pDFTsFBMC)

# Add zeros in OFDM so that the plot looks nicer
Pt_OFDM = np.vstack( (0,Pt_OFDM,0) )

# Calculate time axis
t_OFDM = np.arange(ofdm.nr_samples+2) * ofdm.phy_dt - ofdm.phy_dt
t_FBMC = np.arange(fbmc.nr_samples) * fbmc.phy_dt - ((fbmc.imp_overlapping_factor-0.5) * fbmc.imp_FFTsize/2) * fbmc.phy_dt

# Calculate frequency axis
f_OFDM = np.arange(ofdm.nr_samples) * ofdm.phy_sampling_rate / ofdm.nr_samples - L*F + 0.5*F
f_FBMC = np.arange(fbmc.nr_samples) * fbmc.phy_sampling_rate / fbmc.nr_samples - L*F + 0.5*F

# Calculate PAPR CCDF 
def ccdf(x):
    xs = np.sort(x, axis=0)
    ys = np.arange(len(xs), 0, -1)/float(len(xs))
    return xs, ys
CCDF_PAPR_OFDM_xaxis, CCDF_PAPR_OFDM           = ccdf( 10*np.log10(PAPR_OFDM.reshape(np.size(PAPR_OFDM),1)) )
CCDF_PAPR_SCFDMA_xaxis, CCDF_PAPR_SCFDMA       = ccdf( 10*np.log10(PAPR_SCFDMA.reshape(np.size(PAPR_SCFDMA),1)) )
CCDF_PAPR_FBMC_xaxis, CCDF_PAPR_FBMC           = ccdf( 10*np.log10(PAPR_FBMC.reshape(np.size(PAPR_FBMC),1)) )
CCDF_PAPR_pDFTsFBMC_xaxis, CCDF_PAPR_pDFTsFBMC = ccdf( 10*np.log10(PAPR_pDFTsFBMC.reshape(np.size(PAPR_pDFTsFBMC),1)) )



# =============================================================================
# ============================== Plot Stuff ===================================
# =============================================================================

print("============== Bit rate ================")
print("OFDM:       " + "%8d Bits/s" % ((bitstream_OFDM.size) / (ofdm.nr_symbols_in_time * ofdm.phy_time_spacing)) )
print("SCFDMA:     " + "%8d Bits/s" % ((bitstream_OFDM.size) / (ofdm.nr_symbols_in_time * ofdm.phy_time_spacing)))
print("OFDMnoCP:   " + "%8d Bits/s" % ((bitstream_OFDMnoCP.size) / (ofdm_noCP.nr_symbols_in_time * ofdm_noCP.phy_time_spacing)) )
print("SCFDMAnoCP: " + "%8d Bits/s" % ((bitstream_OFDMnoCP.size) / (ofdm_noCP.nr_symbols_in_time * ofdm_noCP.phy_time_spacing)) )
print("FBMC:       " + "%8d Bits/s" % ((bitstream_FBMC.size) / (fbmc.nr_symbols_in_time * fbmc.phy_time_spacing)) )
print("pDFTsFBMC:  " + "%8d Bits/s" % ((bitstream_pDFTsFBMC.size) / (fbmc_trunc.nr_symbols_in_time * fbmc_trunc.phy_time_spacing)) )
print("========================================")


p1 = plt.figure()
plt.semilogy(SNR_dB, np.mean(BER_OFDMnoCP, axis=1),'red', label='OFDM (noCP)')
plt.semilogy(SNR_dB, np.mean(BER_OFDM, axis=1),'black', label='OFDM')
plt.semilogy(SNR_dB, np.mean(BER_FBMC, axis=1),'blue', label='FBMC')
plt.xlabel('Signal-to-Noise Ratio [dB]')
plt.ylabel('Bit Error Ratio')
plt.ylim( (10**(-4),1) )
plt.xlim( (min(SNR_dB), max(SNR_dB)) )
plt.legend()
plt.grid()
plt.title(pdp + ', v=' + str(v_kmh) + 'km/h' )


p2 = plt.figure()
plt.semilogy(SNR_dB,np.mean(BER_SCFDMAnoCP, axis=1),'red', label='SC-FDMA (noCP)')
plt.semilogy(SNR_dB,np.mean(BER_SCFDMA, axis=1),'black', label='SC-FDMA')
plt.semilogy(SNR_dB,np.mean(BER_pDFTsFBMC, axis=1),'blue', label='p-DFT-s FBMC')
plt.xlabel('Signal-to-Noise Ratio [dB]')
plt.ylabel('Bit Error Ratio')
plt.ylim( (10**(-4),1) )
plt.xlim( (min(SNR_dB), max(SNR_dB)) )
plt.legend()
plt.grid()
plt.title(pdp + ', v=' + str(v_kmh) + 'km/h' )


p3 = plt.figure()
plt.plot(t_OFDM*F, Pt_OFDM, 'red')
plt.plot(t_FBMC*F, Pt_FBMC, 'black')
plt.plot(t_FBMC*F, Pt_pDFTsFBMC, 'blue')
plt.xlim( (-0.5, 0.5) )
plt.xlabel('Normalized Time, tF')
plt.ylabel('Average Power')
plt.legend(('OFDM', 'FBMC', 'p-DFT-s FBMC'))


p4 = plt.figure()
plt.plot(f_OFDM/F, 10*np.log10(Pf_OFDM), 'red')
plt.plot(f_FBMC/F, 10*np.log10(Pf_FBMC), 'black')
plt.plot(f_FBMC/F, 10*np.log10(Pf_pDFTsFBMC), 'blue')
plt.ylim( (-60,0) )
plt.xlim( (-5, 20) )
plt.xlabel('Normalized Frequency, f/F')
plt.ylabel('Power Spectral Density')
plt.legend(('OFDM', 'FBMC', 'p-DFT-s FBMC'))
plt.grid()


p5 = plt.figure()
plt.semilogy(CCDF_PAPR_OFDM_xaxis, CCDF_PAPR_OFDM, 'red')
plt.semilogy(CCDF_PAPR_SCFDMA_xaxis, CCDF_PAPR_SCFDMA, 'green')
plt.semilogy(CCDF_PAPR_FBMC_xaxis, CCDF_PAPR_FBMC, 'black')
plt.semilogy(CCDF_PAPR_pDFTsFBMC_xaxis, CCDF_PAPR_pDFTsFBMC, 'blue')
plt.ylim( (10**(-3), 1))
plt.xlabel('Peak-to-Average-Power-Ratio')
plt.ylabel('CCDF')
plt.legend(('OFDM','SC-FDMA', 'FBMC', 'p-DFT-s FBMC'))
plt.grid()
plt.title( str(L) + ' subcarriers, ' + str(QAM_order) + 'QAM')


p6 = plt.figure()
ax = p6.add_subplot(111, projection='3d')
t_3d, f_3d = np.meshgrid( np.linspace(0,(K_FBMC)*1/F/2,K_FBMC), np.linspace(0,(L-1)*F,L))
ax.plot_surface(t_3d/1e-3, f_3d/1e6, (np.abs(h_FBMC)), cmap=cm.coolwarm)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Frequency [MHz]')
ax.set_zlabel('Transfer Function, |H(t,f)|')
ax.set_title('Possible Channel Realization')
plt.show()


if save_png:
    p2.savefig("png/Figure_2.png",bbox_inches='tight')
    p3.savefig("png/Figure_3.png",bbox_inches='tight')
    p4.savefig("png/Figure_4.png",bbox_inches='tight')
    p5.savefig("png/Figure_5.png",bbox_inches='tight')


