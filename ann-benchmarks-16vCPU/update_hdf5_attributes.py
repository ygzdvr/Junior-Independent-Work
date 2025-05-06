import h5py
import os
import sys
import glob
import argparse

def print_attributes(file_path):
    """Print all attributes of an HDF5 file."""
    try:
        with h5py.File(file_path, "r") as f:
            print(f"Attributes for {os.path.basename(file_path)}:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def update_attribute(file_path, attr_name, attr_value):
    """Update a specific attribute in an HDF5 file."""
    try:
        with h5py.File(file_path, "r+") as f:
            old_value = f.attrs.get(attr_name, "Not present")
            f.attrs[attr_name] = attr_value
            print(f"Updated {attr_name} in {os.path.basename(file_path)}:")
            print(f"  Old value: {old_value}")
            print(f"  New value: {attr_value}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def batch_update_attributes(directory, pattern, attr_name, attr_value):
    """Update attributes for all matching files in a directory."""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found matching {search_path}")
        return
    
    print(f"Found {len(files)} files matching pattern '{pattern}' in {directory}")
    for file_path in files:
        update_attribute(file_path, attr_name, attr_value)
    
    print(f"Updated {len(files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage HDF5 file attributes")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Print command
    print_parser = subparsers.add_parser("print", help="Print file attributes")
    print_parser.add_argument("file", help="HDF5 file path")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update a file attribute")
    update_parser.add_argument("file", help="HDF5 file path")
    update_parser.add_argument("name", help="Attribute name")
    update_parser.add_argument("value", help="New attribute value")
    
    # Batch update command
    batch_parser = subparsers.add_parser("batch", help="Update attributes for multiple files")
    batch_parser.add_argument("directory", help="Directory containing HDF5 files")
    batch_parser.add_argument("pattern", help="File pattern (e.g. '*.hdf5')")
    batch_parser.add_argument("name", help="Attribute name")
    batch_parser.add_argument("value", help="New attribute value")
    
    args = parser.parse_args()
    
    if args.command == "print":
        print_attributes(args.file)
    elif args.command == "update":
        update_attribute(args.file, args.name, args.value)
    elif args.command == "batch":
        batch_update_attributes(args.directory, args.pattern, args.name, args.value)
    else:
        parser.print_help() 