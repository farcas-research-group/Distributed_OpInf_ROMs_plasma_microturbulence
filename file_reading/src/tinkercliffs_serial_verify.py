import numpy as np
import os

nx, nky, nz, nv, nw = 96, 16, 16, 32, 8
grid_size = (nx, nky, nz, nv, nw)
N = np.prod(grid_size)

base_dir = "/scratch/jackkelley/experiments_CBC"
input_filename = os.path.join(base_dir, "g1.dat")

data_dtype = np.float32 
accum_dtype = np.complex64 
step_size_bytes = 8 + (2 * N * 4)

if not os.path.exists(input_filename):
    raise FileNotFoundError(f"{input_filename} does not exist")

file_size = os.path.getsize(input_filename)
n_time = file_size // step_size_bytes
remainder = file_size % step_size_bytes

print(f"--- Serial Verification ---")
print(f"File: {input_filename}")
print(f"Detected Time Steps: {n_time}")
if remainder != 0:
    print(f"WARNING: Remainder bytes detected: {remainder}")

total_Qstar_Q = 0.0 + 0.0j

raw_buffer = np.zeros(2 * N, dtype=data_dtype)

with open(input_filename, 'rb') as fstream:
    for t in range(n_time):
        if t % 10 == 0: 
            print(f"Processing step {t}/{n_time}...")

        data_offset = (t * step_size_bytes) + 8
        fstream.seek(data_offset)
        
        fstream.readinto(raw_buffer)
        
        real_part = raw_buffer[0::2]
        imag_part = raw_buffer[1::2]
        
        temp_complex = (real_part + 1j * imag_part).astype(accum_dtype)
        
        step_energy = np.vdot(temp_complex, temp_complex)
    
        total_Qstar_Q += step_energy

print(f"--------------------------------------------------")
print(f"SERIAL Checksum (Q^H * Q): {total_Qstar_Q}")
print(f"--------------------------------------------------")