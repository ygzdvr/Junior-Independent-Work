#!/usr/bin/env python3
"""
Converts saved NumPy data to gnuplot-readable format.
"""

import os
import numpy as np
import json
import glob

def convert_m_value_data(m_value):
    """Convert data for a specific M value to gnuplot-readable format"""
    data_dir = os.path.join("output", "m_analysis", "data", f"m_{m_value}")
    output_dir = os.path.join("m_analysis", "gnuplot_data", f"m_{m_value}")
    
    if not os.path.exists(data_dir):
        print(f"No data found for M={m_value}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    try:
        with open(os.path.join(data_dir, "parameters.json"), 'r') as f:
            params = json.load(f)
        
        ef_values = params["ef_values"]
        dimension = params["dimension"]
        
        # Load sizes
        sizes = np.load(os.path.join(data_dir, "sizes.npy"))
        
        # Write sizes to a file
        with open(os.path.join(output_dir, "sizes.dat"), 'w') as f:
            for size in sizes:
                f.write(f"{size}\n")
        
        # Write parameters to a file
        with open(os.path.join(output_dir, "parameters.dat"), 'w') as f:
            f.write(f"M {m_value}\n")
            f.write(f"DIM {dimension}\n")
            f.write(f"EF_VALUES {' '.join(map(str, ef_values))}\n")
        
        # Process construction times
        construction_data = []
        for ef in ef_values:
            file_path = os.path.join(data_dir, f"construction_ef_{ef}.npy")
            if os.path.exists(file_path):
                times = np.load(file_path)
                construction_data.append((ef, times))
        
        # Write construction data
        with open(os.path.join(output_dir, "construction.dat"), 'w') as f:
            f.write("# size")
            for ef, _ in construction_data:
                f.write(f" ef_{ef}")
            f.write("\n")
            
            for i, size in enumerate(sizes):
                f.write(f"{size}")
                for _, times in construction_data:
                    if i < len(times):
                        f.write(f" {times[i]}")
                    else:
                        f.write(" 0")
                f.write("\n")
        
        # Process search times
        search_data = []
        for ef in ef_values:
            file_path = os.path.join(data_dir, f"search_ef_{ef}.npy")
            if os.path.exists(file_path):
                times = np.load(file_path)
                search_data.append((ef, times))
        
        # Write search data
        with open(os.path.join(output_dir, "search.dat"), 'w') as f:
            f.write("# size")
            for ef, _ in search_data:
                f.write(f" ef_{ef}")
            f.write("\n")
            
            for i, size in enumerate(sizes):
                f.write(f"{size}")
                for _, times in search_data:
                    if i < len(times):
                        f.write(f" {times[i]}")
                    else:
                        f.write(" 0")
                f.write("\n")
        
        # Process insertion times
        insertion_data = []
        for ef in ef_values:
            file_path = os.path.join(data_dir, f"insertion_ef_{ef}.npy")
            if os.path.exists(file_path):
                times = np.load(file_path)
                insertion_data.append((ef, times))
        
        # Write insertion data
        with open(os.path.join(output_dir, "insertion.dat"), 'w') as f:
            f.write("# size")
            for ef, _ in insertion_data:
                f.write(f" ef_{ef}")
            f.write("\n")
            
            for i, size in enumerate(sizes):
                f.write(f"{size}")
                for _, times in insertion_data:
                    if i < len(times):
                        f.write(f" {times[i]}")
                    else:
                        f.write(" 0")
                f.write("\n")
        
        # Write total insertion time data
        with open(os.path.join(output_dir, "total_insertion.dat"), 'w') as f:
            f.write("# size")
            for ef, _ in insertion_data:
                f.write(f" ef_{ef}")
            f.write("\n")
            
            for i, size in enumerate(sizes):
                f.write(f"{size}")
                for _, times in insertion_data:
                    if i < len(times):
                        # Total insertion time = time per insertion * size
                        f.write(f" {times[i] * size}")
                    else:
                        f.write(" 0")
                f.write("\n")
        
        print(f"Converted data for M={m_value} to gnuplot format in {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error converting data for M={m_value}: {e}")
        return False

def convert_all_data():
    """Convert all saved M value data to gnuplot format"""
    # Create output directory
    os.makedirs(os.path.join("m_analysis", "gnuplot_data"), exist_ok=True)
    
    # Find all M directories
    data_dir = os.path.join("output", "m_analysis", "data")
    if not os.path.exists(data_dir):
        print("No data directory found.")
        return
    
    m_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("m_")]
    
    if not m_dirs:
        print("No M value data found.")
        return
    
    print(f"Found {len(m_dirs)} M values to convert.")
    
    # Convert each M value
    for m_dir in m_dirs:
        m_value = int(m_dir.split("_")[1])
        convert_m_value_data(m_value)
    
    print("All data converted to gnuplot format.")

if __name__ == "__main__":
    convert_all_data() 