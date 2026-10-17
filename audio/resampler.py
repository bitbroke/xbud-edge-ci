import numpy as np
from scipy.signal import resample_poly

class StatefulResampler:
    def __init__(self, in_rate: int, out_rate: int, channels: int = 1):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        
        # Determine rational factor
        gcd = np.gcd(in_rate, out_rate)
        self.up = out_rate // gcd
        self.down = in_rate // gcd
        
        # Scipy's resample_poly uses a Kaiser windowed sinc filter.
        # Its half-length is roughly 10 * max(up, down) by default.
        # We need an overlap buffer at least this large to avoid boundary transients.
        self.overlap_size = 100 * max(self.up, self.down)
        self.buffer = np.zeros((self.overlap_size, self.channels), dtype=np.float32)
        self.first_chunk = True

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
            
        if self.in_rate == self.out_rate:
            # Maybe mixdown to mono if requested, but assume caller handles it or we do it here
            if self.channels == 1 and chunk.shape[1] > 1:
                return chunk.mean(axis=1, keepdims=True)
            return chunk

        # Mix down to target channels before resampling if necessary
        if self.channels == 1 and chunk.shape[1] > 1:
            chunk = chunk.mean(axis=1, keepdims=True)

        if self.first_chunk:
            resampled = resample_poly(chunk, self.up, self.down, axis=0)
            if len(chunk) > self.overlap_size:
                self.buffer = chunk[-self.overlap_size:]
            else:
                # In edge cases where chunk is very small, pad it
                self.buffer = np.pad(chunk, ((self.overlap_size - len(chunk), 0), (0, 0)))
            self.first_chunk = False
            return resampled
            
        # Overlap save: prepend previous buffer
        concat = np.vstack((self.buffer, chunk))
        resampled = resample_poly(concat, self.up, self.down, axis=0)
        
        # Discard the part corresponding to the overlap buffer
        # Output length corresponding to overlap = overlap_size * up / down
        discard_samples = (self.overlap_size * self.up) // self.down
        valid_resampled = resampled[discard_samples:]
        
        # Update buffer
        if len(chunk) >= self.overlap_size:
            self.buffer = chunk[-self.overlap_size:]
        else:
            self.buffer = concat[-self.overlap_size:]
            
        return valid_resampled
