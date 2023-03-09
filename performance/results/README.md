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
600|54000|100000|4|30.1045|20.735|1.45|0.3625

- The `Num Processes` column is the number of processes used to run the simulation: 
	- In `merged_serial_cuda_results.csv` it is the size of the blocks of the kernel that computes the diffusion.
	- In `merged_serial_mpi_results.csv` it is the number of processes launched to perform the diffusion.

- The `Time Serial` column is the time it took to run the simulation using the serial implementation.

- The `Time Parallel` column is the time it took to run the simulation using the parallel implementation.

- The `Speedup` column is the speedup obtained by using the parallel implementation. The formula used is: `serial time / parallel time`.

- The `Efficiency` column is the efficiency obtained by using the parallel implementation.
	- For the MPI results, the formula used is: `speedup / num process`.
	- For the CUDA results, the number of processes is computed with: `num threads per SM * num of SMs`. <br>The number of threads per SM is computed with: `max number of threads per SM / block size`.

The script also outputs two `png` file with the speedup and efficiency graphs, one for the cuda implementation and one for the MPI one.

The graphs saved are made on a subset of the data, just to show it visually:
- `Size` = 600
- `Iterations` = 100000
They show the speedup and efficiency of the implementations as a function of the number of particles and number of processes.

Example of the speedup graph of the MPI implementation:
<br>
<br>
![Speedup MPI](/performance/results/analyzed/graphs/mpi/speedup_mpi_600_100000.png)
