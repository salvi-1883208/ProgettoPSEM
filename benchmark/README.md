# Benchmarking the performance of the various implementations (2D only)


The `benchmark.py` script in this folder is used to do a benchmark of the various 2D implementations.
<br>

It builds the commands to launch the executables and then launches them with `subprocess.run()`.
<br>

The results are then saved in a `implementation_results.csv` file in the `results` folder.
<br>

In the `executables` folder there are the executables for the various implementations.
<br>

To run the benchmark, simply run `python3 benchmark.py` in this folder.
<br>
