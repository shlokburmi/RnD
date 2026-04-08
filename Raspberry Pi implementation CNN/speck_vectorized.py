import numpy as np
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class VectorizedSPECK:
    """
    Optimized NumPy Vectorized SPECK128 implementation.
    Processes blocks in parallel using NumPy bitwise operations.
    """
    
    def __init__(self, key_bytes, key_size=256):
        self.mod_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
        self.word_size = 64
        
        # Determine number of rounds and words
        if key_size == 128:
            self.rounds = 32
            m = 2
        elif key_size == 192:
            self.rounds = 33
            m = 3
        else: # 256
            self.rounds = 34
            m = 4
            
        key_bytes = key_bytes.ljust(m * 8, b'\x00')[:m*8]
        words = [int.from_bytes(key_bytes[i:i+8], 'little') for i in range(0, m*8, 8)]
        
        # Key expansion (scalar)
        k = words[0]
        l = words[1:]
        
        self.round_keys = [np.uint64(k)]
        for i in range(self.rounds - 1):
            # i is used directly in round function
            l_val = l[i % (m-1)]
            
            # Round function for key expansion
            new_l = (self._ror_scalar(l_val, 8) + k) & self.mod_mask
            new_l ^= np.uint64(i)
            
            # Update key words
            if m > 2:
                # For m > 2, we update the list of l words
                l.append(new_l)
            else:
                l[0] = new_l
                
            k = (self._rol_scalar(k, 3) ^ new_l) & self.mod_mask
            self.round_keys.append(np.uint64(k))

    def _ror_scalar(self, x, n):
        return ((x >> np.uint64(n)) | (x << np.uint64(64 - n))) & self.mod_mask

    def _rol_scalar(self, x, n):
        return ((x << np.uint64(n)) | (x >> np.uint64(64 - n))) & self.mod_mask

    def encrypt(self, data):
        """
        Memory-efficient vectorized encryption for Raspberry Pi 4.
        Uses in-place NumPy operations to minimize RAM spikes on 1GB models.
        """
        # Padding
        pad_len = (16 - len(data) % 16) % 16
        if pad_len == 0: pad_len = 16
        # Use bytearray for efficient padding
        data = bytearray(data)
        data.extend([pad_len] * pad_len)
        
        # View as uint64 without copying (extremely important for RPi 1GB RAM)
        data_view = np.frombuffer(data, dtype="<u8")
        
        # Use slices to avoid copies
        x = data_view[0::2].copy() # We still need a copy of slices because they are non-contiguous
        y = data_view[1::2].copy()
        
        # Optimize round loop for ARM NEON throughput
        for rk in self.round_keys:
            # ROR(x, 8) in-place
            x = (x >> np.uint64(8)) | (x << np.uint64(56))
            x += y
            x &= self.mod_mask # Ensure 64-bit wrap
            x ^= rk
            
            # ROL(y, 3) in-place
            y = (y << np.uint64(3)) | (y >> np.uint64(61))
            y ^= x
            
        # Re-interleave efficiently
        result = np.empty(len(data_view), dtype="<u8")
        result[0::2] = x
        result[1::2] = y
        return result.tobytes()

    def decrypt(self, data):
        """
        Memory-efficient vectorized decryption for Raspberry Pi 4.
        """
        data_view = np.frombuffer(data, dtype="<u8")
        
        # Slices require a copy for manipulation, but we keep them as small as possible
        x = data_view[0::2].copy()
        y = data_view[1::2].copy()
        
        for rk in reversed(self.round_keys):
            y ^= x
            # ROR(y, 3) in-place
            y = (y >> np.uint64(3)) | (y << np.uint64(61))
            
            x ^= rk
            # Subtraction handles overflow natively in uint64
            x = (x - y) & self.mod_mask
            
            # ROR(x, -8) -> ROL(x, 8)
            x = (x << np.uint64(8)) | (x >> np.uint64(56))
            
        result = np.empty(len(data_view), dtype="<u8")
        result[0::2] = x
        result[1::2] = y
        
        res_bytes = result.tobytes()
        pad_len = res_bytes[-1]
        
        # Robust padding removal
        if 1 <= pad_len <= 16:
            return res_bytes[:-pad_len]
        return res_bytes
