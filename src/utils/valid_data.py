import os
import h5py
import json
from tqdm import tqdm
from collections import Counter
from hydra import initialize, initialize_config_module, compose
from omegaconf import OmegaConf
from collections import Counter
import hydra

# Function to validate, filter, and save files
def validate_and_filter_files(prefix_file, data_file, directory):
    # Extract expected count from the file name (e.g., "50" from "_50_data.h5")
    try:
        expected_count = int(data_file.split("_")[1])
    except (IndexError, ValueError):
        print(f"Unable to extract expected count from {data_file}. Skipping.")
        return

    # Construct full paths
    prefix_path = os.path.join(directory, prefix_file)
    data_path = os.path.join(directory, data_file)

    # Load the prefix data
    with open(prefix_path, "r") as f:
        prefix_data = [json.loads(line) for line in tqdm(f, desc=f"Loading {prefix_file}")]

    # Load the HDF5 data
    with h5py.File(data_path, "r") as h5f:
        dataset_name = list(h5f.keys())[0]  # Assume a single dataset key
        data = h5f[dataset_name][:]  # Load all data into memory

    # Count occurrences of each tree_encoded
    tree_encoded_list = [json.dumps(entry["tree_encoded"]) for entry in prefix_data]
    counts = Counter(tree_encoded_list)

    # Filter entries
    valid_indices = []
    for idx, entry in enumerate(prefix_data):
        tree_encoded_str = json.dumps(entry["tree_encoded"])
        count = counts[tree_encoded_str]

        # Determine the valid number of entries to keep
        if count >= expected_count:
            n = count // expected_count  # Number of complete expected_count blocks
            remaining_limit = n * expected_count
            if tree_encoded_list[:remaining_limit].count(tree_encoded_str) < remaining_limit:
                valid_indices.append(idx)

    # If no valid indices remain, delete the files
    if not valid_indices:
        print(f"No valid samples in {prefix_file} and {data_file}. Deleting files.")
        os.remove(prefix_path)
        os.remove(data_path)
        return

    # Filter prefix data and HDF5 data
    filtered_prefix_data = [prefix_data[i] for i in valid_indices]
    filtered_h5_data = data[valid_indices]

    # Save the filtered data
    with open(prefix_path, "w") as f:
        for entry in tqdm(filtered_prefix_data, desc=f"Saving filtered {prefix_file}"):
            json.dump(entry, f)
            f.write("\n")

    with h5py.File(data_path, "w") as h5f_filtered:
        h5f_filtered.create_dataset(
            dataset_name,
            data=filtered_h5_data,
            track_times=False
        )

    print(f"Validated and filtered files saved:\n{prefix_path}\n{data_path}")


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
            validate_and_filter_files(prefix_file, data_file, root)

if __name__ == '__main__':
    main()