#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>

// Include all the files that were split from the original hnswalg.h
#include "hnsw_types.h"
#include "hnsw_class_declaration.h"
#include "hnsw_constructors.h"
#include "hnsw_utils.h"
#include "hnsw_search.h"
#include "hnsw_elements.h"
#include "hnsw_structure.h"
#include "hnsw_persistence.h" 
#include "hnsw_quantization.h"

// Other necessary includes from the original hnswalg.h might be needed here too,
// depending on what they contained beyond forward declarations.
// E.g., includes for space implementations, distance functions, etc.
#include "hnswlib.h" // Assuming this pulls in space interfaces etc. 