import os
import re
import shutil
import sys
import h5py
import glob

SOURCE_DIR = "results/nytimes-256-angular/10/my-algorithm-base-dimred-experiments"
TARGET_DIRS_MAP = {
    "240": "240D",
    "224": "224D",
    "240_224": "240D-224D",
    "224_240": "240D-224D", # Handle both orders just in case
    "256": "256D",
}

# Regex to find the target_dims part of the filename
dims_pattern = re.compile(r"_target_dims_(\d+(?:_\d+)?)_use_dim_reduction_")

def organize_and_update_results(base_path, dry_run=False):
    source_path = os.path.join(base_path, SOURCE_DIR)

    if not os.path.isdir(source_path):
        print(f"Error: Source directory not found: {source_path}")
        sys.exit(1)

    print(f"Scanning directory: {source_path}")

    moved_count = 0
    updated_count = 0
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
                    new_algo_name = f"my-algorithm-base-dimred-{target_subdir_name}"
                    
                    # Create target subdirectory if it doesn't exist
                    if not dry_run:
                        os.makedirs(target_dir, exist_ok=True)

                    dest_path = os.path.join(target_dir, filename)

                    print(f"File: '{filename}'")
                    print(f"  - Move to: '{target_subdir_name}/'")
                    print(f"  - Update algo attribute to: '{new_algo_name}'")
                    
                    if not dry_run:
                        # Update the algo attribute before moving
                        try:
                            with h5py.File(file_path, "r+") as f:
                                old_algo = f.attrs.get("algo", "Not set")
                                print(f"  - Old algo value: '{old_algo}'")
                                f.attrs["algo"] = new_algo_name
                                updated_count += 1
                        except Exception as e:
                            print(f"  - Error updating attributes: {e}")
                        
                        # Move the file
                        shutil.move(file_path, dest_path)
                        moved_count += 1
                else:
                    print(f"Warning: Could not map dimensions '{dims_str}' from file '{filename}' to a target directory. Skipping.")
                    skipped_count += 1
            else:
                print(f"Warning: Could not parse target dimensions from file '{filename}'. Skipping.")
                skipped_count += 1

    print(f"\nOrganization complete.")
    if dry_run:
        print("This was a dry run - no files were actually moved or modified.")
    else:
        print(f"Moved {moved_count} files.")
        print(f"Updated {updated_count} files.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (check warnings above).")

if __name__ == "__main__":
    # Parse command line arguments
    dry_run = False
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run = True
        print("Running in dry-run mode - no files will be modified or moved.")
    
    # Assuming the script is run from the ann-benchmarks root directory
    current_working_directory = os.getcwd()
    organize_and_update_results(current_working_directory, dry_run) 