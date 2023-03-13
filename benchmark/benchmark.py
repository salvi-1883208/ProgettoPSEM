# program to benchmark different version of DLA
import subprocess
import math
import time

# build the command
def build_command(executable, size, coverage, iterations, seed, processors, rand):
    executable = "executables/" + executable
    if "serial" in executable:
        command = "./{} {} {} {} {} {} {}".format(
            executable,
            size,
            math.ceil(size * size * (coverage / 100)),
            iterations,
            *seed,
            rand
        )
    elif "cuda" in executable:
        command = "./{} {} {} {} {} {} {} {}".format(
            executable,
            size,
            math.ceil((size * size * (coverage / 100)) / processors),
            iterations,
            *seed,
            processors,
            rand
        )
    else:
        command = "mpirun -np {} ./{} {} {} {} {} {} {}".format(
            processors,
            executable,
            size,
            math.ceil(size * size * (coverage / 100)),
            iterations,
            *seed,
            rand
        )
    return (command, coverage)


# run the command
def run_benchmark(command):
    print(command)
    output = subprocess.check_output(command, shell=True)
    return output.decode("utf-8")


# clean the output
def clean_output(o, c, coverage):
    type = "s" if "serial" in c else "c" if "cuda" in c else "m"
    num_process = c.split()[2] if "mpi" in c else c.split()[6] if "cuda" in c else 1
    c = c[c.find("/") :]
    size = c.split()[1]
    particles = (
        c.split()[2] if "cuda" not in c else (int(size) * int(size) * (coverage / 100))
    )
    iterations = c.split()[3]
    skipped = int(o[o.find("skipped") + 8 : o.find(".", o.find("skipped"))])
    time = float(o[o.find("in seconds:") + 12 : o.find("in seconds:") + 20])
    # type size particles iterations skipped time num_process
    output = "{} {} {} {} {} {} {}".format(
        type, size, int(particles), iterations, skipped, time, num_process
    )
    return output


# the different versions of the program
# executables = ["serial", "cuda", "mpi"]
input = input("Enter the version to benchmark (c: cuda, s: serial, m: mpi): ")

executable = "cuda" if input == "c" else "serial" if input == "s" else "mpi"

# in pixels
sizes = [200, 400, 600]

# percentage of the grid
coverages = [5, 10, 15]

# number of iterations
iterations = [1000, 10000, 100000]

# seeds for the random number generator
rands = [1234, 5678, 9012, 3456]

# number of processors
processors_cuda = [139, 256, 777, 1024]
processors_mpi = [2, 4, 5, 6]

# starting seed
seed = (-1, -1)  # middle of the grid

# file name
filename = executable + "_" + "results.csv"

results = {}

for size in sizes:
    for coverage in coverages:
        for iteration in iterations:
            for rand in rands:
                if "serial" in executable:
                    command = build_command(
                        executable, size, coverage, iteration, seed, 1, rand
                    )
                    output = run_benchmark(command[0])
                    results[command] = clean_output(output, command[0], command[1])
                elif "cuda" in executable:
                    for processor in processors_cuda:
                        command = build_command(
                            executable,
                            size,
                            coverage,
                            iteration,
                            seed,
                            processor,
                            rand,
                        )
                        output = run_benchmark(command[0])
                        results[command] = clean_output(output, command[0], command[1])
                else:
                    for processor in processors_mpi:
                        command = build_command(
                            executable,
                            size,
                            coverage,
                            iteration,
                            seed,
                            processor,
                            rand,
                        )
                        output = run_benchmark(command[0])
                        results[command] = clean_output(output, command[0], command[1])

# write the results to a file
with open("results/" + filename, "w") as f:
    f.write("type size particles iterations skipped time num_process\n")
    for command in results:
        f.write(results[command] + "\n")
