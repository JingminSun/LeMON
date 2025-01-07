import os
import h5py
import json
from tqdm import tqdm
from hydra import initialize, initialize_config_module, compose
from omegaconf import OmegaConf
from collections import Counter
import hydra

# Function to sort files
def sort_files(prefix_file, data_file, directory):
    # Construct full paths
    prefix_path = os.path.join(directory, prefix_file)
    data_path = os.path.join(directory, data_file)
    sorted_prefix_path = os.path.join(directory, prefix_file.replace(".prefix", "_sorted.prefix"))
    sorted_data_path = os.path.join(directory, data_file.replace("_data.h5", "_sorted_data.h5"))

    # Load the prefix data
    with open(prefix_path, "r") as f:
        prefix_data = [json.loads(line) for line in tqdm(f, desc=f"Loading {prefix_file}")]

    # Load the HDF5 data
    with h5py.File(data_path, "r") as h5f:
        dataset_name = list(h5f.keys())[0]  # Assume a single dataset key
        data = h5f[dataset_name][:]  # Load all data into memory

    # Sort both datasets
    indexed_prefix_data = list(enumerate(prefix_data))
    sorted_data = sorted(
        tqdm(indexed_prefix_data, desc=f"Sorting {prefix_file}"), key=lambda x: tuple(x[1]["tree_encoded"])
    )
    sorted_indices = [index for index, _ in sorted_data]
    sorted_prefix_data = [entry for _, entry in sorted_data]
    sorted_h5_data = data[sorted_indices]

    # Save the sorted prefix data
    with open(sorted_prefix_path, "w") as f:
        for entry in tqdm(sorted_prefix_data, desc=f"Saving {sorted_prefix_path}"):
            json.dump(entry, f)
            f.write("\n")

    # Save the sorted HDF5 data
    with h5py.File(sorted_data_path, "w") as h5f_sorted:
        h5f_sorted.create_dataset(
            dataset_name,
            data=sorted_h5_data,
            track_times=False
        )

    # Delete original files
    os.remove(prefix_path)
    os.remove(data_path)

    # Rename sorted files to original names
    os.rename(sorted_prefix_path, prefix_path)
    os.rename(sorted_data_path, data_path)

    print(f"Files sorted and replaced:\n{prefix_path}\n{data_path}")



@hydra.main(config_path=None, config_name=None)
def main(cfg):
    base_directory = cfg.base_directory

    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_directory):
        # Find .prefix and corresponding _data.h5 files in the current directory
        prefix_files = [f for f in files if f.endswith(".prefix")]
        data_files = [f for f in files if f.endswith("_data.h5")]

        # Pair them based on common prefixes in their filenames
        pairs = []
        for prefix_file in prefix_files:
            base_name = prefix_file.replace(".prefix", "")
            matching_data_file = f"{base_name}_data.h5"
            if matching_data_file in data_files:
                pairs.append((prefix_file, matching_data_file))

        # Process each pair
        for prefix_file, data_file in pairs:
            print(f"Processing pair in {root}: {prefix_file}, {data_file}")
            sort_files(prefix_file, data_file, root)


if __name__ == "__main__":
    main()
