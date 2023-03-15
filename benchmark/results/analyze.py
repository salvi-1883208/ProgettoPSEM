import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the CSV files
serial_data = pd.read_csv("serial_results.csv", sep=" ")
mpi_data = pd.read_csv("mpi_results.csv", sep=" ")
cuda_data = pd.read_csv("cuda_results.csv", sep=" ")


# Group the first three columns and calculate the mean of 'time' and 'skipped_column'
grouped_serial = (
    serial_data.groupby(["size", "particles", "iterations", "num_process"])[
        ["time", "skipped"]
    ]
    .mean()
    .reset_index()
)

# Repeat each row in grouped_serial three times
grouped_serial = grouped_serial.loc[grouped_serial.index.repeat(4)].reset_index(
    drop=True
)

# Rename the 'time' column to 'mean_time' and the 'skipped_column' column to 'mean_skipped'
grouped_serial = grouped_serial.rename(
    columns={"time": "mean_time", "skipped": "mean_skipped"}
)

# Round the mean time and skipped columns to four decimal places
grouped_serial["mean_time"] = grouped_serial["mean_time"].round(decimals=4)
grouped_serial["mean_skipped"] = grouped_serial["mean_skipped"].round(decimals=4)

# Write the grouped data to a new CSV file
# grouped_serial.to_csv("analyzed/grouped_serial_results.csv", index=False)

# Group the first three columns and calculate the mean of 'time' and 'skipped_column'
grouped_mpi = (
    mpi_data.groupby(["size", "particles", "iterations", "num_process"])[
        ["time", "skipped"]
    ]
    .mean()
    .reset_index()
)

# Rename the 'time' column to 'mean_time' and the 'skipped_column' column to 'mean_skipped'
grouped_mpi = grouped_mpi.rename(
    columns={"time": "mean_time", "skipped": "mean_skipped"}
)

# Round the mean time and skipped columns to four decimal places
grouped_mpi["mean_time"] = grouped_mpi["mean_time"].round(decimals=4)
grouped_mpi["mean_skipped"] = grouped_mpi["mean_skipped"].round(decimals=4)

# Write the grouped data to a new CSV file
# grouped_mpi.to_csv("analyzed/grouped_mpi_results.csv", index=False)

# Group the first columns and calculate the mean of 'time' and 'skipped_column'
grouped_cuda = (
    cuda_data.groupby(["size", "particles", "iterations", "num_process"])[
        ["time", "skipped"]
    ]
    .mean()
    .reset_index()
)

# Rename the 'time' column to 'mean_time' and the 'skipped_column' column to 'mean_skipped'
grouped_cuda = grouped_cuda.rename(
    columns={"time": "mean_time", "skipped": "mean_skipped"}
)

# Round the mean time and skipped columns to four decimal places
grouped_cuda["mean_time"] = grouped_cuda["mean_time"].round(decimals=4)
grouped_cuda["mean_skipped"] = grouped_cuda["mean_skipped"].round(decimals=4)

# Write the grouped data to a new CSV file
# grouped_cuda.to_csv("analyzed/grouped_cuda_results.csv", index=False)


# merge the grouped serial and cuda dataframes
merged_serial_cuda = pd.merge(
    grouped_serial,
    grouped_cuda,
    on=["size", "iterations", "particles"],
    suffixes=("_serial", "_cuda"),
)

# remove the duplicated rows
merged_serial_cuda = merged_serial_cuda.drop_duplicates()

# Drop the columns that are not needed
merged_serial_cuda = merged_serial_cuda.drop(columns=["num_process_serial"])
merged_serial_cuda = merged_serial_cuda.drop(columns=["mean_skipped_serial"])
merged_serial_cuda = merged_serial_cuda.drop(columns=["mean_skipped_cuda"])

# move the 'particles_cuda' column to the second position
merged_serial_cuda.insert(1, "particles", merged_serial_cuda.pop("particles"))

# move num_process_cuda to the 4 position
merged_serial_cuda.insert(
    3, "num_process_cuda", merged_serial_cuda.pop("num_process_cuda")
)

# max number of threads per block: 1024
# max threads per multiprocessor: 2048
# number of multiprocessors: 20
# max threads per block: 1024
# max blocks per multiprocessor: 32

# number of blocks per multiprocessor = max number of threads per multiprocessor / number of threads per block


# compute the speedup
merged_serial_cuda["speedup"] = (
    merged_serial_cuda["mean_time_serial"] / merged_serial_cuda["mean_time_cuda"]
).round(decimals=2)

# compute the number of blocks in a multiprocessor
num_blocks = (2048 / merged_serial_cuda["num_process_cuda"]).round(decimals=0)

# compute the number of threads per multiprocessor
num_threads = (num_blocks * merged_serial_cuda["num_process_cuda"]).round(decimals=0)

# compute the efficiency
merged_serial_cuda["efficiency"] = (
    merged_serial_cuda["speedup"] / (num_threads * 20)
).round(decimals=4)

# Write the merged data to a new CSV file
merged_serial_cuda.to_csv("analyzed/merged_serial_cuda_results.csv", index=False)

# merge the grouped serial and mpi dataframes
merged_serial_mpi = pd.merge(
    grouped_serial,
    grouped_mpi,
    on=[
        "size",
        "iterations",
        "particles",
    ],
    suffixes=("_serial", "_mpi"),
)

# remove the duplicated rows
merged_serial_mpi = merged_serial_mpi.drop_duplicates()

# Drop the columns that are not needed
merged_serial_mpi = merged_serial_mpi.drop(columns=["num_process_serial"])
merged_serial_mpi = merged_serial_mpi.drop(columns=["mean_skipped_serial"])
merged_serial_mpi = merged_serial_mpi.drop(columns=["mean_skipped_mpi"])

# move the 'particle' column to the second position
merged_serial_mpi.insert(1, "particles", merged_serial_mpi.pop("particles"))

# move num_process_mpi to the 4 position
merged_serial_mpi.insert(3, "num_process_mpi", merged_serial_mpi.pop("num_process_mpi"))

# compute the speedup
merged_serial_mpi["speedup"] = (
    merged_serial_mpi["mean_time_serial"] / merged_serial_mpi["mean_time_mpi"]
).round(decimals=2)

# compute the efficiency
merged_serial_mpi["efficiency"] = (
    merged_serial_mpi["speedup"] / merged_serial_mpi["num_process_mpi"]
).round(decimals=2)

# Write the merged data to a new CSV file
merged_serial_mpi.to_csv("analyzed/merged_serial_mpi_results.csv", index=False)

# get the different number of iterations, size and number of processes
iterations = merged_serial_cuda["iterations"].unique()
size = merged_serial_cuda["size"].unique()
num_process = merged_serial_cuda["num_process_cuda"].unique()

se = ["speedup", "efficiency"]
# create a histogram for each combination of iterations and size (cuda speedup and efficiency)
for v in se:
    for iteration in iterations:
        for s in size:
            # create a new dataframe with the data for the current combination
            histogram_serial_cuda = merged_serial_cuda[
                (merged_serial_cuda["iterations"] == iteration)
                & (merged_serial_cuda["size"] == s)
            ].copy()

            particles = histogram_serial_cuda["particles"].unique()

            x = np.arange(len(particles))  # the label locations
            width = 0.15  # the width of the bars

            fig, ax = plt.subplots(layout="constrained")
            multiplier = 0

            # for each number of particles create a group of bars
            for i in range(len(particles)):
                # for each number of processes create a bar
                for j in range(len(num_process)):
                    # get the data for the current number of particles and processes
                    data = histogram_serial_cuda[
                        (histogram_serial_cuda["particles"] == particles[i])
                        & (histogram_serial_cuda["num_process_cuda"] == num_process[j])
                    ]
                    # get the speedup
                    speedup = data[v].values[0]
                    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
                    offset = width * multiplier
                    # plot the bar without duplicating the label and color
                    if i == 0:
                        rect = ax.bar(
                            x[i] - width + j * width + offset,
                            speedup,
                            width,
                            label=f"{num_process[j]} threads",
                            color=colors[j],
                        )
                    else:
                        rect = ax.bar(
                            x[i] - width + j * width + offset,
                            speedup,
                            width,
                            color=colors[j],
                        )
                    ax.bar_label(rect, padding=3)
                multiplier += 1
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(v.capitalize())
            ax.set_xlabel("Number of particles")
            ax.set_title(
                f"{v.capitalize()} for {s}x{s} grid and {iteration} iterations (CUDA)"
            )
            ax.set_xticks([width - width / 2, width * 8.2])
            ax.set_xticklabels(particles)
            ax.set_ylim(0, 1.3 * max(histogram_serial_cuda[v]))
            ax.legend(ncol=len(num_process), loc="upper left")
            fig.set_size_inches(10, 6)
            # save the plot
            plt.savefig(f"analyzed/graphs/cuda/{v}/{s}_{iteration}.png")
            plt.close()


# get the different number of iterations, size and number of processes
iterations = merged_serial_mpi["iterations"].unique()
size = merged_serial_mpi["size"].unique()
num_process = merged_serial_mpi["num_process_mpi"].unique()

# create a histogram for each combination of iterations and size (mpi speedup and efficiency)
for v in se:
    for iteration in iterations:
        for s in size:
            # create a new dataframe with the data for the current combination
            histogram_serial_mpi = merged_serial_mpi[
                (merged_serial_mpi["iterations"] == iteration)
                & (merged_serial_mpi["size"] == s)
            ].copy()

            particles = histogram_serial_mpi["particles"].unique()

            x = np.arange(len(particles))
            width = 0.15

            fig, ax = plt.subplots(layout="constrained")
            multiplier = 0

            # for each number of particles create a group of bars
            for i in range(len(particles)):
                # for each number of processes create a bar
                for j in range(len(num_process)):
                    # get the data for the current number of particles and processes
                    data = histogram_serial_mpi[
                        (histogram_serial_mpi["particles"] == particles[i])
                        & (histogram_serial_mpi["num_process_mpi"] == num_process[j])
                    ]
                    # get the speedup
                    speedup = data[v].values[0]
                    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
                    offset = width * multiplier
                    # plot the bar without duplicating the label and color
                    if i == 0:
                        rect = ax.bar(
                            x[i] - width + j * width + offset,
                            speedup,
                            width,
                            label=f"{num_process[j]} processes",
                            color=colors[j],
                        )
                    else:
                        rect = ax.bar(
                            x[i] - width + j * width + offset,
                            speedup,
                            width,
                            color=colors[j],
                        )
                    ax.bar_label(rect, padding=3)
                multiplier += 1
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(v.capitalize())
            ax.set_xlabel("Number of particles")
            ax.set_title(
                f"{v.capitalize()} for {s}x{s} grid and {iteration} iterations (MPI)"
            )
            ax.set_xticks([width - width / 2, width * 8.2])
            ax.set_xticklabels(particles)
            ax.set_ylim(0, 1.3 * max(histogram_serial_mpi[v]))
            ax.legend(ncol=len(num_process), loc="upper left")
            fig.set_size_inches(10, 6)
            # save the plot
            plt.savefig(f"analyzed/graphs/mpi/{v}/{s}_{iteration}.png")
            plt.close()
