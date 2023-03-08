import pandas as pd

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

# Group the first three columns and calculate the mean of 'time' and 'skipped_column'
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
    on=[
        "size",
        "iterations",
    ],
    suffixes=("_serial", "_cuda"),
)

# Drop the columns that are not needed
merged_serial_cuda = merged_serial_cuda.drop(columns=["num_process_serial"])
merged_serial_cuda = merged_serial_cuda.drop(columns=["particles_serial"])
merged_serial_cuda = merged_serial_cuda.drop(columns=["mean_skipped_serial"])
merged_serial_cuda = merged_serial_cuda.drop(columns=["mean_skipped_cuda"])

# move the 'particles_cuda' column to the second position and multiply it by "num_process_cuda"
merged_serial_cuda.insert(
    1,
    "particles_cuda",
    merged_serial_cuda.pop("particles_cuda") * merged_serial_cuda["num_process_cuda"],
)

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

# in my case, the number of parallel threads is always 40960

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
    ],
    suffixes=("_serial", "_mpi"),
)

# Drop the columns that are not needed
merged_serial_mpi = merged_serial_mpi.drop(columns=["num_process_serial"])
merged_serial_mpi = merged_serial_mpi.drop(columns=["particles_serial"])
merged_serial_mpi = merged_serial_mpi.drop(columns=["mean_skipped_serial"])
merged_serial_mpi = merged_serial_mpi.drop(columns=["mean_skipped_mpi"])

# move the 'particles_mpi' column to the second position and multiply it by "num_process_mpi"
merged_serial_mpi.insert(
    1,
    "particles_mpi",
    merged_serial_mpi.pop("particles_mpi") * merged_serial_mpi["num_process_mpi"],
)

# move num_process_mpi to the 4 position
merged_serial_mpi.insert(3, "num_process_mpi", merged_serial_mpi.pop("num_process_mpi"))

# compute the speedup
merged_serial_mpi["speedup"] = (
    merged_serial_mpi["mean_time_serial"] / merged_serial_mpi["mean_time_mpi"]
).round(decimals=2)

# compute the efficiency
merged_serial_mpi["efficiency"] = (
    merged_serial_mpi["speedup"] / merged_serial_mpi["num_process_mpi"]
).round(decimals=4)

# Write the merged data to a new CSV file
merged_serial_mpi.to_csv("analyzed/merged_serial_mpi_results.csv", index=False)
