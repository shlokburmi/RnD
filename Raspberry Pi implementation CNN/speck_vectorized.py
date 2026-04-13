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
            l_val = l[i % (m-1)]
            new_l = (self._ror_scalar(l_val, 8) + k) & self.mod_mask
            new_l ^= np.uint64(i)
            if m > 2:
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
        """
        # Ensure data is a multiple of 16 for block cipher (128-bit blocks)
        pad_len = (16 - (len(data) % 16)) % 16
        if pad_len > 0:
            data = bytearray(data)
            data.extend([pad_len] * pad_len)
        elif len(data) == 0:
            data = bytearray(b'\x10' * 16)
        
        # View as uint64
        data_view = np.frombuffer(data, dtype="<u8")
        
        # Slit into left/right words
        x = data_view[0::2].copy()
        y = data_view[1::2].copy()
        
        for rk in self.round_keys:
            x = (x >> np.uint64(8)) | (x << np.uint64(56))
            x += y
            x &= self.mod_mask
            x ^= rk
            y = (y << np.uint64(3)) | (y >> np.uint64(61))
            y ^= x
            
        result = np.empty(len(data_view), dtype="<u8")
        result[0::2] = x
        result[1::2] = y
        return result.tobytes()

    def decrypt(self, data):
        """
        Memory-efficient vectorized decryption for Raspberry Pi 4.
        """
        # Buffer alignment check for uint64
        rem = len(data) % 8
        if rem != 0:
            data = bytearray(data)
            data.extend([0] * (8 - rem))

        data_view = np.frombuffer(data, dtype="<u8")
        
        # In case data length is not a multiple of 16 (incomplete block)
        if len(data_view) % 2 != 0:
            data_view = np.append(data_view, np.uint64(0))

        x = data_view[0::2].copy()
        y = data_view[1::2].copy()
        
        for rk in reversed(self.round_keys):
            y ^= x
            y = (y >> np.uint64(3)) | (y << np.uint64(61))
            x ^= rk
            x = (x - y) & self.mod_mask
            x = (x << np.uint64(8)) | (x >> np.uint64(56))
            
        result = np.empty(len(data_view), dtype="<u8")
        result[0::2] = x
        result[1::2] = y
        
        res_bytes = result.tobytes()
        if len(res_bytes) == 0: return b''
        pad_len = res_bytes[-1]
        
        if 1 <= pad_len <= 16:
            return res_bytes[:-pad_len]
        return res_bytes
