# Results of the benchmarks

## Results
The results of the benchmarks done are saved as `csv` files.

The `csv` files have the following structure: 
| Type | Size | Particles | Iterations | Skipped | Time | Num Processes|
|------|------|-----------|------------|---------|------|--------------|
|m |400 |16000 |100000 |929 |3.853909 |6

- The `Type` column is the type of the implementation used. It can be either `m` for the MPI implementation, `c` for the cuda implementation or `s` for the serial implementation.

- The `Size` column is the size of the grid used.

- The `Particles` column is the number of particles launched.

- The `Iterations` column is the maximum number of iterations permitted for each particle.

- The `Skipped` column is the number of particles that were skipped because they reached the maximum number of iterations.

- The `Time` column is the time it took to run the single simulation.

- The `Num Processes` column is the number of processes used to run the simulation.


## Analysis
The analysis of the results is done using the `analyze.py` script.

The script takes as input the `csv` files and outputs two `csv` file with the following structure:

| Size | Particles | Iterations | Num Processes|Time Serial | Time Parallel |  Speedup | Efficiency |
|------|-----------|------------|--------------|------------|---------------|----------|------------|
700|49000|100000|6|50.8853|26.6812|1.91|0.32

- The `Num Processes` column is the number of processes used to run the simulation: 
	- In `merged_serial_cuda_results.csv` it is the size of the blocks of the kernel that computes the diffusion.
	- In `merged_serial_mpi_results.csv` it is the number of processes launched to perform the diffusion.

- The `Time Serial` column is the time it took to run the simulation using the serial implementation.

- The `Time Parallel` column is the time it took to run the simulation using the parallel implementation.

- The `Speedup` column is the speedup obtained by using the parallel implementation. The formula used is: `serial time / parallel time`.

- The `Efficiency` column is the efficiency obtained by using the parallel implementation.
	- For the MPI results, the formula used is: `speedup / num process`.
	- For the CUDA results, the number of processes is computed with: `num threads per SM * num of SMs`. <br>The number of threads per SM is computed with: `max number of threads per SM / block size`.

## Graphs
The script also generates graphs of the speedup and efficiency obtained by using the parallel implementations.

The graps are saved with the following structure: `graphs/{type}/{graph}/{size}_{iterations}.png`, where:
- `type` is the type of the implementation used. It can be either `mpi` for the MPI implementation or `cuda` for the cuda implementation.
- `graph` is the type of the graph. It can be either `speedup` or `efficiency`.
- `size` is the size of the grid used.
- `iterations` is the maximum number of iterations permitted for each particle.

##


Example of a graph of the speedup obtained by using the MPI implementation with a grid of size 700 and 100000 iterations per particle:
<br>
<br>
![Speedup MPI](/benchmark/results/analyzed/graphs/mpi/speedup/700_100000.png)
