import numpy as np
import os
import re
import argparse
from mpi4py import MPI
from typing import Tuple, Optional, Any

def parse_num_steps_from_parameters(param_path: str) -> int:
    with open(param_path, "r") as f:
        for line in f:
            if "number of computed time steps" in line.lower():
                m = re.findall(r"(-?\d+)", line)
                if not m:
                    raise ValueError(f"Found key line but no integer: {line.strip()}")
                return int(m[-1])
    raise FileNotFoundError(f"Could not find 'number of computed time steps' in {param_path}")

def distribute_work(N: int, rank: int, size: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    base_size = N // size
    remainder = N % size
    counts = np.full(size, base_size, dtype=int)
    counts[-1] += remainder
    displs = np.zeros(size, dtype=int)
    displs[1:] = np.cumsum(counts)[:-1]
    return counts, displs, counts[rank], displs[rank]

def read_distribution_parallel(
    comm: MPI.Comm,
    input_filename: str,
    param_path: str,
    grid_size: Tuple[int, ...],
    embeddings: int = 1,
    precision: type = np.float64,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    
    rank = comm.rank
    size = comm.size
    
    if precision == np.float64:
        complex_dtype = np.complex128
        item_size = 8
    elif precision == np.float32:
        complex_dtype = np.complex64
        item_size = 4
    else:
        raise ValueError("Precision must be np.float64 or np.float32")

    N = np.prod(grid_size)

    # Distribute indices
    counts_N, displs_N, size_per_rank, start_idx_complex = distribute_work(N, rank, size)
    print(f"[Rank {rank:02d}] Start Index: {start_idx_complex:9d} | Work Size: {size_per_rank:7d} items")

    # Find the number of time steps
    n_time = 0
    step_size_bytes = 8 + (2 * N * item_size)  # time (8) + data (2*N*size)

    full_times = None
    if rank == 0:
        full_times = np.empty(n_time, dtype=precision) 

    if rank == 0:
        if not os.path.exists(input_filename):
            print(f"Error: File {input_filename} not found.")
            n_time = -1
        else:
            file_bytes = os.path.getsize(input_filename)
            n_time = file_bytes // step_size_bytes
            print(f"Calculated Time Steps (n_time): {n_time}")
    
    n_time = comm.bcast(n_time, root=0)
    if n_time <= 0:
        comm.Abort(1)

    # Check embedding constraints
    n_cols = n_time - embeddings + 1
    if n_cols <= 0:
        if rank == 0:
            print(f"Error: Number of embeddings ({embeddings}) must be less than or equal to total time steps ({n_time}).")
        comm.Abort(1)

    # Allocate local matrix Q_i (rows=space, cols=time)
    local_Q = np.zeros((size_per_rank, n_time), dtype=complex_dtype)
    
    # Buffers for reading
    local_data_block_float = np.zeros(size_per_rank * 2, dtype=precision)

    # Read loop (Fill local_Q column by column)
    if rank == 0:
        print(f"Opening file {input_filename}...")
        
    with open(input_filename, "rb") as fstream:
        for t in range(n_time):
            if rank == 0 and (t % 10 == 0 or t == n_time - 1):
                print(f"Reading time step {t+1}/{n_time}...")
            
            step_offset = t * step_size_bytes
            
            # Read time
            if rank == 0:
                fstream.seek(step_offset)
                full_times[t] = np.fromfile(fstream, dtype=np.float64, count=1)[0]

            # Read data
            rank_offset_bytes = step_offset + 8 + (start_idx_complex * 2 * item_size)
            fstream.seek(rank_offset_bytes)
            fstream.readinto(local_data_block_float.data)

            # De-interleave and store in column t of local_Q
            local_real = local_data_block_float[0::2]
            local_imag = local_data_block_float[1::2]
            local_Q[:, t] = local_real + 1j * local_imag

    # Construct the time delay embedded matrix H_i
    if rank == 0:
        print(f"Constructing delay matrix with {embeddings} embeddings (New time dimension: {n_cols})")
    
    # local_H will have shape (size_per_rank * embeddings, n_cols)
    local_H = np.zeros((size_per_rank * embeddings, n_cols), dtype=complex_dtype)
    for e in range(embeddings):
        local_H[e * size_per_rank : (e + 1) * size_per_rank, :] = local_Q[:, e : e + n_cols]

    # Find D_i = H_i^H * H_i
    print(f"[Rank {rank:02d}] Computing local matrix product")
    local_D = np.matmul(local_H.conj().T, local_H)

    # Reduce to get global D = sum(D_i)
    if rank == 0:
        global_D = np.zeros_like(local_D)
    else:
        global_D = None

    comm.Reduce(local_D, global_D, op=MPI.SUM, root=0)
    
    # Truncate times to match the new embedded column dimension
    if rank == 0:
        full_times = full_times[:n_cols]

    # Output
    if rank == 0:
        print("-" * 40)
        print(f"Embedded Matrix D Shape: {global_D.shape}")
        
        # Calculate trace
        trace_val = np.trace(global_D)
        print(f"Trace(D) (Checksum): {trace_val}")
        
        # Verify is hermitian
        is_hermitian = np.allclose(global_D, global_D.conj().T)
        print(f"Matrix is Hermitian: {is_hermitian}")
        print("-" * 40)
    
    return None, full_times

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--sim_index", type=int, required=True)
    parser.add_argument("-b", "--base_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-g", "--grid", nargs='+', type=int, required=False)
    parser.add_argument("-p", "--precision", type=str, choices=["float64", "float32"], default="float64")
    parser.add_argument("-e", "--embeddings", type=int, default=1, help="Number of time delay embeddings")
    return parser.parse_args()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    args = None
    grid_size = None

    if rank == 0:
        args = parse_arguments()
        if args.grid is None:
            sim_dir = args.base_dir 
            param_path = os.path.join(sim_dir, "parameters.dat")
            grid_size = parse_num_steps_from_parameters(param_path=param_path)
        else:
            grid_size = tuple(args.grid)

    args = comm.bcast(args, root=0)
    grid_size = comm.bcast(grid_size, root=0)
    
    data_precision = np.float64 if args.precision == "float64" else np.float32
    param_path = os.path.join(args.base_dir, "parameters.dat")
    input_filename = os.path.join(args.base_dir, "g1.dat")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Processing Grid: {grid_size}")

    read_distribution_parallel(
        comm,
        input_filename,
        param_path,
        grid_size,
        embeddings=args.embeddings,
        precision=data_precision
    )

if __name__ == "__main__":
    main()