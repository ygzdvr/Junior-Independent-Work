import h5py
import sys
import os

def print_hdf5_attributes(filename):
    """
    Print all attributes of an HDF5 file.
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    if not filename.endswith('.hdf5'):
        print(f"Warning: {filename} doesn't have .hdf5 extension")
    
    try:
        with h5py.File(filename, "r") as f:
            print(f"Attributes for {filename}:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error opening {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_hdf5_attributes.py <hdf5_file_path>")
        sys.exit(1)
    
    print_hdf5_attributes(sys.argv[1]) 