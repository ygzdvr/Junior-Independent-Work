import os
import re
import shutil
import sys

SOURCE_DIR = "results/nytimes-256-angular/10/my-algorithm-base-dimred-experiments"
TARGET_DIRS_MAP = {
    "240": "240D",
    "224": "224D",
    "240_224": "240D-224D",
    "224_240": "240D-224D", # Handle both orders just in case
}

# Regex to find the target_dims part of the filename
# It looks for _target_dims_ followed by one or two numbers (separated by _)
# and captures the number(s) before _use_dim_reduction_
# Example: ..._target_dims_240_224_use_dim_reduction_... -> captures "240_224"
# Example: ..._target_dims_240_use_dim_reduction_... -> captures "240"
dims_pattern = re.compile(r"_target_dims_(\d+(?:_\d+)?)_use_dim_reduction_")

def organize_results(base_path):
    source_path = os.path.join(base_path, SOURCE_DIR)

    if not os.path.isdir(source_path):
        print(f"Error: Source directory not found: {source_path}")
        sys.exit(1)

    print(f"Scanning directory: {source_path}")

    moved_count = 0
    skipped_count = 0

    for filename in os.listdir(source_path):
        if filename.endswith(".hdf5"):
            file_path = os.path.join(source_path, filename)

            # Skip if it's not a file (e.g., a directory)
            if not os.path.isfile(file_path):
                continue

            match = dims_pattern.search(filename)
            if match:
                dims_str = match.group(1)
                if dims_str in TARGET_DIRS_MAP:
                    target_subdir_name = TARGET_DIRS_MAP[dims_str]
                    target_dir = os.path.join(source_path, target_subdir_name)

                    # Create target subdirectory if it doesn't exist
                    os.makedirs(target_dir, exist_ok=True)

                    dest_path = os.path.join(target_dir, filename)

                    # Move the file
                    print(f"Moving '{filename}' to '{target_subdir_name}/'")
                    shutil.move(file_path, dest_path)
                    moved_count += 1
                else:
                    print(f"Warning: Could not map dimensions '{dims_str}' from file '{filename}' to a target directory. Skipping.")
                    skipped_count += 1
            else:
                print(f"Warning: Could not parse target dimensions from file '{filename}'. Skipping.")
                skipped_count += 1

    print(f"\nOrganization complete.")
    print(f"Moved {moved_count} files.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (check warnings above).")

if __name__ == "__main__":
    # Assuming the script is run from the ann-benchmarks root directory
    current_working_directory = os.getcwd()
    organize_results(current_working_directory) 