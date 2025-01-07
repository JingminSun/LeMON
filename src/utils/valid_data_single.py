def validate_and_filter_specific_pair(prefix_file_path, data_file_path, expected_count):
    import os
    import h5py
    import json
    from tqdm import tqdm
    from collections import Counter

    # Check if the files exist
    if not os.path.exists(prefix_file_path) or not os.path.exists(data_file_path):
        print(f"Error: Files not found:\n{prefix_file_path}\n{data_file_path}")
        return

    # Load the prefix data
    with open(prefix_file_path, "r") as f:
        prefix_data = [json.loads(line) for line in tqdm(f, desc=f"Loading {prefix_file_path}")]

    # Load the HDF5 data
    with h5py.File(data_file_path, "r") as h5f:
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
        print(f"No valid samples in {prefix_file_path} and {data_file_path}. Deleting files.")
        os.remove(prefix_file_path)
        os.remove(data_file_path)
        return

    # Filter prefix data and HDF5 data
    filtered_prefix_data = [prefix_data[i] for i in valid_indices]
    filtered_h5_data = data[valid_indices]

    # Save the filtered data
    with open(prefix_file_path, "w") as f:
        for entry in tqdm(filtered_prefix_data, desc=f"Saving filtered {prefix_file_path}"):
            json.dump(entry, f)
            f.write("\n")

    with h5py.File(data_file_path, "w") as h5f_filtered:
        h5f_filtered.create_dataset(
            dataset_name,
            data=filtered_h5_data,
            track_times=False
        )

    print(f"Validated and filtered files saved:\n{prefix_file_path}\n{data_file_path}")

if __name__ == '__main__':

    prefix_file_path = "/home/shared/prosepde/diff_bistablereact_1D/diff_bistablereact_1D_50.prefix"
    data_file_path = "/home/shared/prosepde/diff_bistablereact_1D/diff_bistablereact_1D_50_data.h5"
    expected_count = 50  # Adjust this as per the file name
    validate_and_filter_specific_pair(prefix_file_path, data_file_path, expected_count)
