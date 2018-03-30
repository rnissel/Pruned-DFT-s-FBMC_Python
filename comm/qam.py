# =============================================================================
# ========================== (c) Ronald Nissel ================================
# ======================== First version: 29.03.2018 ==========================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


class QAM: 
    """ 
    QAM(m) 
    
    Transforms a bitstream into QAM symbols and vice versa. LTE compliant. 
    
    Parameters
    ----------
    m : Signal constellation order. Can be m = 4, 16, 64, 256, 1024, 4096, ... 
   
    Usage
    ----------        
    - Transform a binary stream (array consisting of 0 and 1) into QAM symbols with ".convert_bits2symbols( bits )"
    
    - Convert symbols back to a binary stream with the method ".convert_symbols2bits( symbols )"
    
    """
    def __init__(self, qam_order):   
        bits_per_symbol = int(np.log2(qam_order)) 
        
        # Gray coded QAM mapping (LTE standard)
        iq = 2 * np.arange(np.sqrt(qam_order)) - np.sqrt(qam_order) + 1
        q_rep, i_rep = np.meshgrid(iq,iq)       
        symbols = i_rep.reshape(qam_order) + 1j * q_rep.reshape(qam_order)       
        symbols = symbols/np.sqrt(np.mean(np.abs(symbols)**2))         
        a = int(np.sqrt(qam_order)/2)
        bitmapping_atom = np.hstack((np.ones(a), np.zeros(a))) 
        for i in range(int(bits_per_symbol/2-1)):
            if i == 0:
                BitTemp = bitmapping_atom
            else:
                BitTemp = bitmapping_atom[-1,:]               
            bitmapping_atom = np.vstack((bitmapping_atom, np.hstack((BitTemp[::2],BitTemp[::-2]))))
        bitmapping_atom = bitmapping_atom.T    
        bit_mapping = np.zeros((qam_order, bits_per_symbol))
        for x_iq in iq:
            index_i = np.nonzero(i_rep.reshape(qam_order) == x_iq)
            index_q = np.nonzero(q_rep.reshape(qam_order) == x_iq)
            
            if qam_order == 4:
                bit_mapping[index_i,1] = bitmapping_atom
                bit_mapping[index_q,0] = bitmapping_atom
            else:
                bit_mapping[index_i,1::2] = bitmapping_atom
                bit_mapping[index_q,::2] = bitmapping_atom
        bin2dec = np.sum(bit_mapping * 2**np.arange(bits_per_symbol), axis=1, dtype=int)
             
        self.bit_mapping = bit_mapping[np.argsort(bin2dec),:]
        self.symbol_mapping = symbols[np.argsort(bin2dec)]        
        self.bits_per_symbol = bits_per_symbol
        self.qam_order = qam_order
        
    def convert_bits2symbols(self, bitstream):
        """
        Converts a bit stream to the corresponding QAM symbols (gray coded according to the LTE standard)
    
        Input
        ----------
        bitstream : Array consisting of 1 and 0. The size must be a multiple of log2(m), with m the modulation order used for the QAM object. 
    
        Returns
        -------
        out : Complex valued array, representing QAM symbols
        
        """
        bitstream = bitstream.reshape(int(np.size(bitstream)/self.bits_per_symbol), self.bits_per_symbol) 
        return self.symbol_mapping[np.sum(bitstream * 2**np.arange(self.bits_per_symbol), axis=1, dtype=int)]
       
    def convert_symbols2bits(self, symbols):
        """
        Quantization of complex valued symbols to the nearest QAM signal conestllation point and conversion to binary code
    
        Input
        ----------
        symbols : Complex valued array
    
        Returns
        -------
        out : Array consisting of 1 and 0
        
        """        
        distance_symbols_to_constellation = np.abs(symbols.reshape(np.size(symbols), 1, order='F') - self.symbol_mapping)
        return self.bit_mapping[np.argmin(distance_symbols_to_constellation, axis=1),:]
    
    def plot_signal_constellation(self):
        """
        Plot the QAM signal constellation, including the bit mapping
        
        """        
        for i in range(np.size(self.symbol_mapping)):
            plt.plot(np.real(self.symbol_mapping), np.imag(self.symbol_mapping),' bo')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.text(np.real(self.symbol_mapping[i]), np.imag(self.symbol_mapping[i])-0.02, 
                     np.array2string(self.bit_mapping[i,:]).replace('[','').replace(']','').replace('.','').replace(' ',''),
                     horizontalalignment='center',
                      verticalalignment='top', 
                     )
            plt.plot([0,0], [-1,1], color=(0,0,0), linewidth=0.5)
            plt.plot([-1,1], [0,0], color=(0,0,0), linewidth=0.5)
    
    
    
