import numpy as np
import os 
import re

nx0, nky0, nz0, nv0, nw0 = 3, 1, 168, 32, 8  # n_kx x n_z x n_v x n_mu grid size
data_size = (nx0, nz0, nv0, nw0)
N = np.prod(data_size)

base_dir = "/global/cfs/cdirs/m3586/parametric_ETG_ROM/training_folders"

# --------- helpers ----------
def parse_num_steps_from_parameters(param_path: str) -> int:
    """
    Search parameters.dat for a line containing 'number of computed time steps'
    and return the integer found on that line. Flexible to spaces, '=', ':'.
    """
    with open(param_path, "r") as f:
        for line in f:
            if "number of computed time steps" in line.lower():
                # Extract the last integer on the line
                m = re.findall(r"(-?\d+)", line)
                if not m:
                    raise ValueError(f"Found the key line but no integer: {line.strip()}")
                return int(m[-1])
    raise FileNotFoundError(
        f"Could not find 'number of computed time steps' in {param_path}"
    )

# how many training cases to read (there's 84 sims in the full directory)
# just read one for now
for idx in range(1, 2):
    sim_dir = os.path.join(base_dir, f"ETG_sim_{idx}")
    param_path = os.path.join(sim_dir, "parameters.dat")
    num_steps = parse_num_steps_from_parameters(param_path)
    n_time = (num_steps + 1)

    input_filename = os.path.join(base_dir, f"ETG_sim_{idx}", "g1.dat")
    output_raw_data_filename = os.path.join(base_dir, f"ETG_sim_{idx}", "g1_copy.npy")
    output_times_filename    = os.path.join(base_dir, f"ETG_sim_{idx}", "g1_times_copy.npy")    

    times = np.empty(n_time, dtype=np.float64)
    data = np.empty(data_size + (n_time,), dtype=np.complex128, order='F')

    with open(input_filename, 'rb') as fstream:
        for t in range(n_time):
            offset = t * (2 * N + 1)
            position_time = offset * 8  # 8 bytes for double-precision, 4 for single
            fstream.seek(position_time)
            times[t] = np.fromfile(fstream, dtype=np.float64, count=1)[0] # np.float64 for double-precision, np.float32 for single

            position_data = (offset + 1) * 8 # 8 bytes for double-precision, 4 for single
            fstream.seek(position_data)
            data_block = np.fromfile(fstream, dtype=np.float64, count=2 * N) # np.float64 for double-precision, np.float32 for single

            data_real_flat = data_block[0::2]
            data_imag_flat = data_block[1::2]

            data_real = data_real_flat.reshape(data_size, order='F')
            data_imag = data_imag_flat.reshape(data_size, order='F')

            data[..., t] = data_real + 1j * data_imag

    data_X = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])

    Qstar_Q_serial = np.vdot(data_X, data_X).real
    print(f"SERIAL Checksum (Q^H * Q): {Qstar_Q_serial}")

    # these are already saved so im commenting these lines out
    # np.save(output_raw_data_filename, data_X)
    # np.save(output_times_filename, times)

    print(f"[ok] sim {idx}: wrote {n_time} time slices")
