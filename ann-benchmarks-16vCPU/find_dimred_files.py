import os
import re
import sys
import glob

# Base results directory
RESULTS_DIR = "results"

# Regex to find the target_dims part of the filename
dims_pattern = re.compile(r"_target_dims_(\d+(?:_\d+)?)_use_dim_reduction_")

def find_dimred_files(base_path):
    """
    Find all HDF5 files containing target_dims in their filename.
    Returns a dict of dimensions to lists of files.
    """
    results_path = os.path.join(base_path, RESULTS_DIR)
    
    if not os.path.isdir(results_path):
        print(f"Error: Results directory not found: {results_path}")
        sys.exit(1)
    
    # Find all HDF5 files recursively
    all_hdf5_files = glob.glob(os.path.join(results_path, "**/*.hdf5"), recursive=True)
    print(f"Found {len(all_hdf5_files)} total HDF5 files in {results_path}")
    
    # Filter for files containing target_dims in their name
    dimred_files = {}
    
    for file_path in all_hdf5_files:
        filename = os.path.basename(file_path)
        match = dims_pattern.search(filename)
        if match:
            dims_str = match.group(1)
            if dims_str not in dimred_files:
                dimred_files[dims_str] = []
            dimred_files[dims_str].append(file_path)
    
    return dimred_files

def print_dimred_summary(dimred_files):
    """Print a summary of dimension reduction files found"""
    if not dimred_files:
        print("No dimension reduction files found!")
        return
    
    print("\nDimension reduction files summary:")
    print("==================================")
    
    for dims, files in sorted(dimred_files.items()):
        print(f"\nTarget dimensions: {dims}")
        print(f"Found {len(files)} files")
        
        # Group files by directory
        by_dir = {}
        for file_path in files:
            dir_path = os.path.dirname(file_path)
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append(os.path.basename(file_path))
        
        # Print directory summaries
        for dir_path, filenames in by_dir.items():
            print(f"\n  Directory: {dir_path}")
            print(f"  Files: {len(filenames)}")
            for filename in sorted(filenames):
                print(f"    - {filename}")

if __name__ == "__main__":
    current_working_directory = os.getcwd()
    dimred_files = find_dimred_files(current_working_directory)
    print_dimred_summary(dimred_files) 