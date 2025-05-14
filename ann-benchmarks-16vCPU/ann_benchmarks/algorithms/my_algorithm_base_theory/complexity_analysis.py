#!/usr/bin/env python3
"""
Complexity Analysis for My Algorithm (based on HNSW)

This script empirically demonstrates the logarithmic time complexity for:
1. Index construction
2. Search operations
3. Insertion operations

The algorithm is based on Hierarchical Navigable Small World (HNSW) graphs.
"""

import hnswlib
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import sys
import multiprocessing as mp
from functools import partial
import json
import random
import psutil  # Add psutil for memory monitoring
import traceback
import gc

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enable debug mode
DEBUG = os.environ.get('DEBUG', '0') == '1'
if DEBUG:
    print("DEBUG MODE ENABLED - More verbose output will be shown")

# Read parameters from environment variables with defaults
M_VALUE = int(os.environ.get('M_VALUE', 16))
EF_CONSTRUCTION = int(os.environ.get('EF_CONSTRUCTION', 200))
EF_SEARCH = int(os.environ.get('EF_SEARCH', 200))
DIMENSION = int(os.environ.get('DIMENSION', 256))
VECTOR_COUNT = int(os.environ.get('VECTOR_COUNT', 1000000))
THREADS = int(os.environ.get('THREADS', mp.cpu_count() - 1))

# Make sure thread count is valid
THREADS = max(1, min(THREADS, mp.cpu_count()))

# Check if we should run extreme scale (for 10M vectors or more)
RUN_EXTREME_SCALE = VECTOR_COUNT >= 1000000

print("Running complexity analysis with the following parameters:")
print(f"  M = {M_VALUE}")
print(f"  EF Construction = {EF_CONSTRUCTION}")
print(f"  EF Search = {EF_SEARCH}")
print(f"  Dimension = {DIMENSION}")
print(f"  Vector Count = {VECTOR_COUNT}")
print(f"  Threads = {THREADS}")
print(f"  Extreme Scale = {RUN_EXTREME_SCALE}")

# Parameters
HIGH_DIM = DIMENSION  # Set to the environment value

def print_memory_usage(label=""):
    """Print current memory usage"""
    if DEBUG:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"MEMORY USAGE ({label}): {mem_info.rss / (1024**3):.2f} GB")

def generate_random_vectors(count, dim):
    """Generate random vectors for testing"""
    print(f"Generating {count} vectors with dimension {dim}")
    print_memory_usage("Before vector generation")
    
    # Use smaller batches for large counts to avoid memory issues
    if count > 1000000:
        batch_size = 500000  # Generate in 500K batches
        vectors = np.empty((count, dim), dtype=np.float32)
        for i in range(0, count, batch_size):
            end_idx = min(i + batch_size, count)
            vectors[i:end_idx] = np.random.random((end_idx - i, dim)).astype(np.float32)
            print(f"Generated batch {i//batch_size + 1}/{(count+batch_size-1)//batch_size}")
            gc.collect()  # Force garbage collection
    else:
        vectors = np.random.random((count, dim)).astype(np.float32)
    
    print_memory_usage("After vector generation")
    return vectors

def run_optimized_single_job(m_value, ef_construction, ef_search, dim=HIGH_DIM, threads=None, max_elements=VECTOR_COUNT):
    """
    Run a single optimized job using all available resources with specific M and EF values.
    
    Args:
        m_value: M parameter for HNSW index
        ef_construction: EF construction parameter
        ef_search: EF search parameter (can be different from ef_construction)
        dim: Vector dimensionality
        threads: Number of threads to use (defaults to all available CPUs)
        max_elements: Maximum number of elements in the index
    """
    try:
        print("=" * 80)
        print(f"RUNNING OPTIMIZED SINGLE JOB ANALYSIS")
        print(f"Parameters: M={m_value}, EF_construction={ef_construction}, EF_search={ef_search}, DIM={dim}")
        print("=" * 80)
        
        # Use all available CPUs if not specified
        if threads is None:
            threads = THREADS
        
        print(f"Using {threads} CPU threads for processing")
        
        # Check available memory and adjust max_elements accordingly
        available_memory = psutil.virtual_memory().available
        bytes_per_vector = dim * 4  # 4 bytes per float32
        
        # Estimate memory needed for the index (very rough estimate)
        # HNSW typically uses ~80-150 bytes per element for the graph structure
        # Plus the actual vectors
        bytes_per_element_in_index = bytes_per_vector + 150 * m_value/16  # Scale with M parameter
        
        # Use at most 30% of available memory for the dataset to leave room for the index
        safe_memory = int(available_memory * 0.3)
        max_elements_by_memory = safe_memory // bytes_per_vector
        
        # Cap the maximum elements based on extreme scale setting or memory
        if max_elements > max_elements_by_memory:
            print(f"WARNING: Requested vector count {max_elements:,} exceeds memory capacity")
            print(f"Reducing to {max_elements_by_memory:,} vectors")
            max_elements = max_elements_by_memory
        
        print(f"Available memory: {available_memory / (1024**3):.2f} GB")
        print(f"Maximum elements based on memory: {max_elements_by_memory:,}")
        print(f"Using {max_elements:,} vectors for analysis")
        
        # Define logarithmically spaced sizes to test
        # Use fewer test points with reduced size range to avoid memory issues
        if max_elements >= 10**7:  # Very large scale
            min_exp, max_exp = 4, int(math.log10(max_elements))
            num_points = 5  # Reduced from 8 to 5 points to save memory
        else:  # More moderate scale
            min_exp, max_exp = 3, int(math.log10(max_elements))
            num_points = 8  # Reduced from 10 to 8 points
        
        # Generate log-spaced points
        exponents = np.linspace(min_exp, max_exp, num_points)
        sizes = np.unique(np.round(10**exponents).astype(int))
        
        print(f"Testing {len(sizes)} different index sizes from {sizes[0]:,} to {sizes[-1]:,}")
        
        # Create output directory
        results_dir = "/output"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate queries - we can generate these upfront since they're small
        query_count = 1000  # Reduced from 3000 to 1000 to save memory
        print(f"Generating {query_count} query vectors...")
        queries = generate_random_vectors(query_count, dim)
        
        # Results arrays
        construction_times = []
        search_times = []
        insertion_times = []
        memory_usages = []
        
        # Track the current state of the index
        current_size = 0
        
        # Run garbage collection before creating the index
        gc.collect()
        
        # Create the index
        print("Creating index...")
        print_memory_usage("Before index creation")
        
        index = hnswlib.Index(space='l2', dim=dim)
        # Set the random seed for hnsw library if possible
        if hasattr(index, 'set_seed'):
            index.set_seed(RANDOM_SEED)
        
        # Try to set the number of threads for hnswlib
        try:
            index.set_num_threads(threads)
            print(f"Set hnswlib to use {threads} threads")
        except Exception as e:
            print(f"Failed to set thread count directly: {str(e)}")
            try:
                hnswlib.set_num_threads(threads)
                print(f"Set hnswlib global threads to {threads}")
            except Exception as e:
                print(f"Failed to set thread count for hnswlib: {str(e)}")
                print("Will use default threading")
        
        # Initialize with our parameters
        # Use a slightly smaller max_elements to avoid memory allocation issues
        index_max_elements = int(max_elements * 1.0)  # Allow 5% overhead for insertions
        print(f"DEBUG: Initializing index with max_elements={index_max_elements}, ef_construction={ef_construction}, M={m_value}")
        index.init_index(max_elements=index_max_elements, ef_construction=ef_construction, M=m_value)
        print(f"DEBUG: Index initialization successful")
        
        print_memory_usage("After index creation")
        
        # Process each target size
        for i, target_size in enumerate(sizes):
            print(f"\n===== Testing size: {target_size:,} vectors ({i+1}/{len(sizes)}) =====")
            print_memory_usage(f"Before size {target_size}")
            
            # How many points to add
            points_to_add = target_size - current_size
            
            if points_to_add <= 0:
                # Skip this size - we've already measured a larger size
                continue
            
            # Memory usage before
            mem_before = psutil.Process().memory_info().rss / (1024**3)  # GB
            
            # Force garbage collection before adding new items
            gc.collect()
            
            if current_size == 0:
                # First size - create index and add vectors in batches to avoid memory issues
                print(f"Constructing index with {target_size:,} vectors...")
                
                # Use smaller batch size for larger dimensions
                # Further reduce batch size for high dimensions
                if dim > 200:
                    batch_size = min(200000, target_size)  # Reduced from 500K to 200K
                else:
                    batch_size = min(500000, target_size)
                
                num_batches = (target_size + batch_size - 1) // batch_size
                
                print(f"Processing in {num_batches} batches of up to {batch_size:,} vectors each")
                
                start_time = time.time()
                
                # Process each batch
                for batch_idx in tqdm(range(num_batches), desc="Adding batches"):
                    # Calculate batch range
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, target_size)
                    batch_count = end_idx - start_idx
                    
                    print(f"Generating batch {batch_idx+1}/{num_batches} with {batch_count} vectors")
                    # Generate batch
                    batch_data = generate_random_vectors(batch_count, dim)
                    
                    # Add batch to index
                    print(f"Adding batch {batch_idx+1} to index")
                    print(f"DEBUG: Beginning add_items for batch {batch_idx+1} with {batch_count} vectors")
                    print(f"DEBUG: Batch data shape: {batch_data.shape}, dtype: {batch_data.dtype}")
                    print(f"DEBUG: First vector sample: {batch_data[0][:5]}...")
                    try:
                        # Try with a reduced number of threads for first batch
                        if batch_idx == 0:
                            # Save current thread setting
                            old_thread_count = index.get_num_threads() if hasattr(index, 'get_num_threads') else None
                            # Use only 1 thread for first batch to avoid potential segfault
                            if hasattr(index, 'set_num_threads'):
                                print(f"DEBUG: Temporarily reducing thread count to 1 for first batch")
                                index.set_num_threads(1)
                        
                        # Add items in smaller chunks to pinpoint segfault
                        chunk_size = 100
                        for chunk_start in range(0, batch_count, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, batch_count)
                            print(f"DEBUG: Adding chunk {chunk_start} to {chunk_end} of batch {batch_idx+1}")
                            # Using add_items instead of the lower-level functions
                            index.add_items(batch_data[chunk_start:chunk_end])
                            print(f"DEBUG: Successfully added chunk {chunk_start} to {chunk_end}")
                        
                        # Restore original thread count after first batch
                        if batch_idx == 0 and old_thread_count is not None and hasattr(index, 'set_num_threads'):
                            print(f"DEBUG: Restoring thread count to {old_thread_count}")
                            index.set_num_threads(old_thread_count)
                            
                    except Exception as e:
                        print(f"ERROR: Failed to add batch {batch_idx+1} to index: {str(e)}")
                        print(f"DEBUG: Error type: {type(e).__name__}")
                        traceback.print_exc()
                        # Save error info
                        with open(f"{results_dir}/add_items_error.json", 'w') as f:
                            error_info = {
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "batch_index": batch_idx,
                                "batch_count": batch_count,
                                "batch_data_shape": batch_data.shape,
                                "batch_data_dtype": str(batch_data.dtype)
                            }
                            json.dump(error_info, f, indent=2)
                        raise
                    
                    print(f"DEBUG: Successfully added batch {batch_idx+1} to index")
                    
                    # Clear batch from memory and force garbage collection
                    print(f"Clearing batch {batch_idx+1} from memory")
                    del batch_data
                    gc.collect()
                    print_memory_usage(f"After batch {batch_idx+1}")
                    
                end_time = time.time()
                
                construction_time = end_time - start_time
                construction_times.append(construction_time)
                print(f"Construction time: {construction_time:.2f} seconds")
                
                current_size = target_size
            else:
                # We already have an index - measure insertion time for additional points
                print(f"Inserting {points_to_add:,} vectors...")
                
                # Process in batches to avoid memory issues
                # Further reduce batch size for high dimensions
                if dim > 200:
                    batch_size = min(200000, points_to_add)  # Reduced from 500K to 200K
                else:
                    batch_size = min(500000, points_to_add)
                
                num_batches = (points_to_add + batch_size - 1) // batch_size
                
                print(f"Processing in {num_batches} batches of up to {batch_size:,} vectors each")
                
                start_time = time.time()
                
                # Process each batch
                for batch_idx in tqdm(range(num_batches), desc="Adding batches"):
                    # Calculate batch range
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, points_to_add)
                    batch_count = end_idx - start_idx
                    
                    print(f"Generating batch {batch_idx+1}/{num_batches} with {batch_count} vectors")
                    # Generate batch
                    batch_data = generate_random_vectors(batch_count, dim)
                    
                    # Add batch to index
                    print(f"Adding batch {batch_idx+1} to index")
                    print(f"DEBUG: Beginning add_items for batch {batch_idx+1} with {batch_count} vectors")
                    print(f"DEBUG: Batch data shape: {batch_data.shape}, dtype: {batch_data.dtype}")
                    try:
                        # Add items in smaller chunks
                        chunk_size = 100
                        for chunk_start in range(0, batch_count, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, batch_count)
                            print(f"DEBUG: Adding chunk {chunk_start} to {chunk_end} of batch {batch_idx+1}")
                            index.add_items(batch_data[chunk_start:chunk_end])
                            print(f"DEBUG: Successfully added chunk {chunk_start} to {chunk_end}")
                    except Exception as e:
                        print(f"ERROR: Failed to add batch {batch_idx+1} to index: {str(e)}")
                        traceback.print_exc()
                        # Save error info
                        with open(f"{results_dir}/add_items_error.json", 'w') as f:
                            error_info = {
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "batch_index": batch_idx,
                                "batch_count": batch_count,
                                "batch_data_shape": batch_data.shape,
                                "batch_data_dtype": str(batch_data.dtype)
                            }
                            json.dump(error_info, f, indent=2)
                        raise
                    
                    # Clear batch from memory and force garbage collection
                    print(f"Clearing batch {batch_idx+1} from memory")
                    del batch_data
                    gc.collect()
                    print_memory_usage(f"After batch {batch_idx+1}")
                    
                end_time = time.time()
                
                # Average insertion time per element
                insertion_time = (end_time - start_time) / points_to_add
                insertion_times.append(insertion_time)
                print(f"Average insertion time: {insertion_time*1000:.3f} ms per vector")
                
                # For construction, estimate what full construction would have taken
                estimated_construction = (construction_times[0] * 
                                         sizes[0] / target_size + 
                                         insertion_time * (target_size - sizes[0]))
                construction_times.append(estimated_construction)
                
                current_size = target_size
            
            # Memory usage after building the index
            mem_after = psutil.Process().memory_info().rss / (1024**3)  # GB
            memory_usages.append(mem_after)
            print(f"Memory usage: {mem_after:.2f} GB (Δ: {mem_after-mem_before:.2f} GB)")
            
            # Measure search time with specified ef_search
            print(f"Measuring search performance with EF_search={ef_search}...")
            index.set_ef(ef_search)
            
            # Run searches
            start_time = time.time()
            for query_idx in tqdm(range(query_count), desc="Searching"):
                try:
                    index.knn_query(queries[query_idx].reshape(1, -1), k=10)
                except Exception as e:
                    print(f"ERROR: Search failed for query {query_idx}: {str(e)}")
                    traceback.print_exc()
                    # Save the last few successful searches
                    break
            end_time = time.time()
            
            # Average time per query
            avg_search_time = (end_time - start_time) / query_count
            search_times.append(avg_search_time)
            print(f"Average search time: {avg_search_time*1000:.3f} ms per query")
            
            # Save intermediate results after each size to avoid losing data
            print("Saving intermediate results...")
            intermediate_results = {
                "parameters": {
                    "m_value": int(m_value),
                    "ef_construction": int(ef_construction),
                    "ef_search": int(ef_search),
                    "dimension": int(dim),
                    "threads": int(threads),
                    "max_elements": int(max_elements),
                    "random_seed": RANDOM_SEED,
                    "timestamp": time.time(),
                    "completed_sizes": i + 1
                },
                "sizes": sizes[:i+1].tolist(),
                "construction_times": construction_times,
                "search_times": search_times,
                "insertion_times": insertion_times,
                "memory_usages": memory_usages
            }
            with open(os.path.join(results_dir, f"intermediate_results_{i}.json"), 'w') as f:
                json.dump(intermediate_results, f, indent=2)
        
        # Save all results
        results = {
            "parameters": {
                "m_value": int(m_value),
                "ef_construction": int(ef_construction),
                "ef_search": int(ef_search),
                "dimension": int(dim),
                "threads": int(threads),
                "max_elements": int(max_elements),
                "random_seed": RANDOM_SEED,
                "timestamp": time.time()
            },
            "sizes": sizes.tolist(),
            "construction_times": construction_times,
            "search_times": search_times,
            "insertion_times": insertion_times,
            "memory_usages": memory_usages
        }
        
        # Save to JSON
        with open(os.path.join(results_dir, "final_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        create_optimized_plots(results_dir, results)
        
        # Clean up index to free memory
        del index
        gc.collect()
        
        print("\nAnalysis complete! Results saved to '{}' directory".format(results_dir))
        return results
    except Exception as e:
        print(f"ERROR: An exception occurred during analysis: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        # Try to save error information
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "parameters": {
                "m_value": int(m_value),
                "ef_construction": int(ef_construction),
                "ef_search": int(ef_search),
                "dimension": int(dim),
                "threads": int(threads),
                "max_elements": int(max_elements)
            },
            "memory": {
                "available": psutil.virtual_memory().available / (1024**3),
                "used": psutil.virtual_memory().used / (1024**3),
                "process": psutil.Process().memory_info().rss / (1024**3)
            }
        }
        
        try:
            with open("/output/error_log.json", 'w') as f:
                json.dump(error_info, f, indent=2)
        except:
            print("Failed to save error information to file")
        
        return None

def create_optimized_plots(results_dir, results):
    """Create comprehensive plots for the optimized single job results"""
    try:
        print("Creating plots...")
        # Extract data from results
        sizes = np.array(results["sizes"])
        construction_times = np.array(results["construction_times"])
        search_times = np.array(results["search_times"])
        insertion_times = np.array(results["insertion_times"]) if "insertion_times" in results else np.array([])
        memory_usages = np.array(results["memory_usages"]) if "memory_usages" in results else np.array([])
        
        params = results["parameters"]
        m_value = params["m_value"]
        ef_construction = params["ef_construction"]
        ef_search = params["ef_search"]
        dim = params["dimension"]
        
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Index Construction Time
        plt.subplot(2, 2, 1)
        plt.plot(sizes, construction_times, 'o-', color='blue', label='Measured')
        
        # Add log curve for comparison
        if len(construction_times) > 0:
            log_curve = [a * n * math.log(n) for n, a in 
                        zip(sizes, [construction_times[-1]/(sizes[-1] * math.log(sizes[-1]))] * len(sizes))]
            plt.plot(sizes, log_curve, '--', color='red', label='O(n log n) reference')
        
        plt.title(f'Index Construction Time (M={m_value}, EF={ef_construction}, DIM={dim})')
        plt.xlabel('Number of vectors')
        plt.ylabel('Time (seconds)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Search Time vs Index Size
        plt.subplot(2, 2, 2)
        plt.plot(sizes, search_times, 'o-', color='green', label='Measured')
        
        # Add log curve for comparison
        if len(search_times) > 0:
            log_curve = [a * math.log(n) for n, a in 
                        zip(sizes, [search_times[-1]/math.log(sizes[-1])] * len(sizes))]
            plt.plot(sizes, log_curve, '--', color='red', label='O(log n) reference')
        
        plt.title(f'Search Time vs Index Size (M={m_value}, EF_search={ef_search}, DIM={dim})')
        plt.xlabel('Index Size (number of vectors)')
        plt.ylabel('Time per query (seconds)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Insertion Time vs Index Size
        plt.subplot(2, 2, 3)
        if len(insertion_times) > 0:
            plt.plot(sizes[1:], insertion_times, 'o-', color='purple', label='Measured')
            
            # Add log curve for comparison
            log_curve = [a * math.log(n) for n, a in 
                        zip(sizes[1:], [insertion_times[-1]/math.log(sizes[-1])] * len(insertion_times))]
            plt.plot(sizes[1:], log_curve, '--', color='red', label='O(log n) reference')
        
        plt.title(f'Insertion Time vs Index Size (M={m_value}, EF={ef_construction}, DIM={dim})')
        plt.xlabel('Index size')
        plt.ylabel('Time per insertion (seconds)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Memory Usage
        plt.subplot(2, 2, 4)
        if len(memory_usages) > 0:
            plt.plot(sizes, memory_usages, 'o-', color='orange', label='Memory usage (GB)')
            
            # Add linear reference line
            linear = [a * n for n, a in zip(sizes, [memory_usages[-1]/sizes[-1]] * len(sizes))]
            plt.plot(sizes, linear, '--', color='red', label='O(n) reference')
        
        plt.title(f'Memory Usage (M={m_value}, EF={ef_construction}, DIM={dim})')
        plt.xlabel('Number of vectors')
        plt.ylabel('Memory usage (GB)')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "performance_plots.png"))
        plt.close()
        
        print("Plots created successfully")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting analysis...")
    print_memory_usage("At start")
    
    try:
        # Run the optimized single job with parameters from environment variables
        results = run_optimized_single_job(
            m_value=M_VALUE,
            ef_construction=EF_CONSTRUCTION,
            ef_search=EF_SEARCH,
            dim=DIMENSION,
            threads=THREADS,
            max_elements=VECTOR_COUNT
        )
        
        # Save results to the output directory for Docker volume mounting
        if results is not None:
            print("Analysis completed successfully!")
            sys.exit(0)
        else:
            print("Analysis failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        
        # Try to save error information
        try:
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "memory": {
                    "available": psutil.virtual_memory().available / (1024**3),
                    "used": psutil.virtual_memory().used / (1024**3),
                    "process": psutil.Process().memory_info().rss / (1024**3)
                }
            }
            
            with open("/output/fatal_error.json", 'w') as f:
                json.dump(error_info, f, indent=2)
        except:
            print("Failed to save error information to file")
        
        sys.exit(1) 