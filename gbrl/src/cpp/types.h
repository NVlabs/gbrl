//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//////////////////////////////////////////////////////////////////////////////

/**
 * @file types.h
 * @brief Core type definitions and data structures for the GBRL library
 * 
 * This header contains all fundamental type definitions, enumerations, and
 * data structures used throughout the Gradient Boosted Reinforcement Learning
 * (GBRL) library, including ensemble metadata, optimization configurations,
 * and device type specifications.
 */

#ifndef TYPES_H
#define TYPES_H

#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

/** @brief Macro for stringifying preprocessor arguments */
#define STRINGIFY(x) #x

/** @brief Converts a macro value to a string */
#define TOSTRING(x) STRINGIFY(x)

/** @brief Initial maximum number of trees in the ensemble (50,000) */
#define INITAL_MAX_TREES 50000

/** @brief Number of trees to add in each batch expansion (25,000) */
#define TREES_BATCH 25000

/** @brief Maximum character size for categorical values */
#define MAX_CHAR_SIZE 128

/** @brief Format specifier for categorical type based on MAX_CHAR_SIZE */
#define CAT_TYPE TOSTRING(MAX_CHAR_SIZE) "s"

/** @brief Forward declaration of Optimizer class */
class Optimizer;

/**
 * @brief Defines a split condition for tree node branching
 * 
 * Contains all information needed to evaluate whether a sample should
 * go to the left or right child of a tree node.
 */
struct splitCondition {
    int feature_idx;                /**< Index of the feature used for splitting */
    float feature_value;            /**< Threshold value for numerical features */
    bool inequality_direction;      /**< Direction of inequality (< or >=) */
    float edge_weight;              /**< Weight associated with this split edge */
    char *categorical_value;        /**< Value for categorical feature comparison */
};

/**
 * @brief Candidate split point to be evaluated during tree growing
 * 
 * Represents a potential split that needs to be scored to determine
 * if it should be used for branching a tree node.
 */
struct splitCandidate {
    int feature_idx;                /**< Index of the feature for this candidate */
    float feature_value;            /**< Threshold value for numerical features */
    char *categorical_value;        /**< Value for categorical features */
};

/**
 * @brief Information about a categorical feature during candidate generation
 * 
 * Stores aggregated statistics for a categorical feature value to help
 * determine promising split candidates.
 */
struct categoryInfo {
    float total_grad_norm;          /**< Sum of gradient norms for this category */
    int cat_count;                  /**< Number of samples in this category */
    int feature_idx;                /**< Index of the categorical feature */
    std::string feature_name;       /**< Name/value of the category */
};

/**
 * @brief Learning rate scheduling functions
 * 
 * Defines how the learning rate changes during optimization.
 */
enum schedulerFunc : uint8_t {
    Const,      /**< Constant learning rate throughout training */
    Linear      /**< Linear decay from initial to final learning rate */
};

/**
 * @brief Optimization algorithms for gradient descent
 */
enum optimizerAlgo : uint8_t {
    SGD,        /**< Stochastic Gradient Descent */
    Adam        /**< Adaptive Moment Estimation (Adam) optimizer */
};

/**
 * @brief Device types for computation
 */
enum deviceType : uint8_t {
    cpu,            /**< CPU-based computation */
    gpu,            /**< GPU-based computation (CUDA) */
    unspecified     /**< Device not yet specified */
};

/**
 * @brief Methods for generating split candidates
 */
enum generatorType : uint8_t {
    Uniform,    /**< Uniformly spaced candidates across feature range */
    Quantile    /**< Quantile-based candidates from data distribution */
};

/**
 * @brief Loss function types for training
 */
enum lossType : uint8_t {
    MultiRMSE,  /**< Multi-output Root Mean Squared Error */
};

/**
 * @brief Scoring functions for evaluating split quality
 */
enum scoreFunc : uint8_t {
    L2,         /**< L2 norm-based scoring */
    Cosine      /**< Cosine similarity-based scoring */
};

/**
 * @brief Null checking status
 */
enum NULL_CHECK : uint8_t {
    NULL_OPT,   /**< Null/optional value */
    VALID       /**< Valid value present */
};

/**
 * @brief Export format for model serialization
 */
enum exportFormat : uint8_t {
    EXP_FLOAT,  /**< Full floating-point precision */
    EXP_FXP8,   /**< 8-bit fixed-point representation */
    EXP_FXP16,  /**< 16-bit fixed-point representation */
};

/**
 * @brief Export type for model compression
 */
enum exportType : uint8_t {
    FULL,       /**< Export complete model structure */
    COMPACT     /**< Export compressed/optimized model */
};

/**
 * @brief Tree growing strategies
 */
enum growPolicy : uint8_t {
    GREEDY,     /**< Greedy growth: best split at each node independently */
    OBLIVIOUS   /**< Oblivious trees: same feature at each depth level */
};

/**
 * @brief Configuration parameters for an optimizer
 * 
 * Contains all settings needed to configure and restore an optimizer's state.
 */
struct optimizerConfig {
    std::string algo;               /**< Algorithm name (SGD, Adam, etc.) */
    std::string scheduler_func;     /**< Learning rate scheduler type */
    float init_lr;                  /**< Initial learning rate */
    float stop_lr;                  /**< Final learning rate (for decay) */
    int start_idx;                  /**< Starting tree index for optimization */
    int stop_idx;                   /**< Stopping tree index for optimization */
    int T;                          /**< Total number of optimization steps */
    float beta_1;                   /**< First moment decay rate (Adam) */
    float beta_2;                   /**< Second moment decay rate (Adam) */
    float eps;                      /**< Numerical stability epsilon (Adam) */
};

/** @brief Alias for vector of floats */
using FloatVector = std::vector<float>;

/** @brief Alias for vector of integers */
using IntVector = std::vector<int>;

/** @brief Alias for vector of booleans */
using BoolVector = std::vector<bool>;

/**
 * @brief Metadata describing the ensemble configuration and current state
 * 
 * Contains all structural and hyperparameter information about the
 * gradient boosted tree ensemble.
 */
struct ensembleMetaData {
    int n_leaves;                   /**< Current number of leaves in ensemble */
    int n_trees;                    /**< Current number of trees in ensemble */
    int max_trees;                  /**< Maximum capacity for trees */
    int max_leaves;                 /**< Maximum capacity for leaves */
    int max_trees_batch;            /**< Max trees to add in one batch */
    int max_leaves_batch;           /**< Max leaves to add in one batch */
    int input_dim;                  /**< Number of input features (numerical) */
    int output_dim;                 /**< Dimension of output/target space */
    int policy_dim;                 /**< Dimension of policy output (for RL) */
    int max_depth;                  /**< Maximum depth of individual trees */
    int min_data_in_leaf;           /**< Minimum samples required per leaf */
    int n_bins;                     /**< Number of bins for split candidates */
    int par_th;                     /**< Parallelization threshold */
    float cv_beta;                  /**< Control variates beta parameter */
    int verbose;                    /**< Verbosity level for logging */
    int batch_size;                 /**< Training batch size */
    bool use_cv;                    /**< Whether to use control variates */
    scoreFunc split_score_func;     /**< Function for scoring splits */
    generatorType generator_type;   /**< Method for generating candidates */
    growPolicy grow_policy;         /**< Tree growing strategy */
    int n_num_features;             /**< Number of numerical features */
    int n_cat_features;             /**< Number of categorical features */
    int iteration;                  /**< Current training iteration */
};

/**
 * @brief Wrapper for data pointers with device location tracking
 * 
 * @tparam T Data type of the held pointer
 * 
 * Encapsulates a data pointer along with information about whether
 * the data resides in CPU or GPU memory.
 */
template<typename T>
struct dataHolder {
    T *data;                        /**< Pointer to the actual data */
    deviceType device;              /**< Device where data is located */
};

/**
 * @brief Complete dataset structure for training and prediction
 * 
 * Contains all data arrays needed for gradient boosting: observations,
 * categorical features, and gradients.
 */
struct dataSet {
    dataHolder<const float> *obs;           /**< Numerical observations */
    dataHolder<const char> *categorical_obs; /**< Categorical observations */
    dataHolder<float> *grads;               /**< Gradient values */
    dataHolder<float> *build_grads;         /**< Gradients for tree building */
    int n_samples;                          /**< Number of data samples */
};

/**
 * @brief Data storage for the entire ensemble of trees
 * 
 * Contains all parameter arrays and tree structure information for
 * the gradient boosted ensemble. Memory layout optimized for both
 * CPU and GPU access patterns.
 */
struct ensembleData {
    float *bias;                    /**< Global bias terms for each output */
    float *feature_weights;         /**< Per-feature importance weights */
#ifdef DEBUG
    int *n_samples;                 /**< Sample counts per leaf (debug only) */
#endif 
    int *tree_indices;              /**< Starting leaf indices for each tree */
    int *depths;                    /**< Depth of each leaf in tree structure */
    float *values;                  /**< Leaf prediction values */
    
    // These arrays support reordering of features for mixed categorical/numerical inputs
    int *feature_mapping;           /**< Maps original feature indices to internal indices (stored for documentation/export) */
    int *reverse_num_feature_mapping;  /**< Maps internal numerical feature indices back to original feature indices (used in computation) */
    int *reverse_cat_feature_mapping;  /**< Maps internal categorical feature indices back to original feature indices (used in computation) */
    // Leaf split condition data
    int* feature_indices;           /**< Feature used at each internal node */
    float* feature_values;          /**< Threshold values for numerical splits */
    float *edge_weights;            /**< Weights for split edges */
    bool* is_numerics;              /**< Whether split is numerical (vs categorical) */
    bool* inequality_directions;    /**< Direction of inequality tests */
    char* categorical_values;       /**< Values for categorical splits */

    bool *mapping_numerics;         /**< Indicates if each original feature is numerical (true) or categorical (false) (stored for documentation/export) */
    
    size_t alloc_data_size;         /**< Total allocated memory size */
};

/**
 * @brief Header for model serialization with version tracking
 * 
 * Ensures compatibility when loading saved models by storing
 * version information.
 */
struct serializationHeader {
    uint16_t major_version;         /**< Major version number */
    uint16_t minor_version;         /**< Minor version number */
    uint16_t patch_version;         /**< Patch version number */
    uint64_t reserved1 = 0;         /**< Reserved for future use */
    uint32_t reserved2 = 0;         /**< Reserved for future use */
};

/**
 * @brief Information about a node's position in the tree structure
 * 
 * Tracks relationships between parent and child nodes during
 * tree traversal and construction.
 */
struct nodeInfo {
    int idx;                        /**< Relative index within current tree */
    int parent_idx;                 /**< Relative index of parent node */
    int depth;                      /**< Depth level in tree (root = 0) */
    bool is_left;                   /**< True if this is a left child */
    bool is_right;                  /**< True if this is a right child */
};

// ============================================================================
// Conversion Functions: String to Enum
// ============================================================================

/** @brief Convert string to scoreFunc enum */
scoreFunc stringToScoreFunc(std::string str);

/** @brief Convert string to exportFormat enum */
exportFormat stringToexportFormat(std::string str);

/** @brief Convert string to exportType enum */
exportType stringToexportType(std::string str);

/** @brief Convert string to generatorType enum */
generatorType stringTogeneratorType(std::string str);

/** @brief Convert string to growPolicy enum */
growPolicy stringTogrowPolicy(std::string str);

/** @brief Convert string to lossType enum */
lossType stringTolossType(std::string str);

/** @brief Convert string to deviceType enum */
deviceType stringTodeviceType(std::string str);

/** @brief Convert string to optimizerAlgo enum */
optimizerAlgo stringToAlgoType(std::string str);

/** @brief Convert string to schedulerFunc enum */
schedulerFunc stringToSchedulerType(std::string str);

// ============================================================================
// Conversion Functions: Enum to String
// ============================================================================

/** @brief Convert scoreFunc enum to string representation */
std::string scoreFuncToString(scoreFunc func);

/** @brief Convert generatorType enum to string representation */
std::string generatorTypeToString(generatorType type);

/** @brief Convert growPolicy enum to string representation */
std::string growPolicyToString(growPolicy type);

/** @brief Convert lossType enum to string representation */
std::string lossTypeToString(lossType type);

/** @brief Convert deviceType enum to string representation */
std::string deviceTypeToString(deviceType type);

/** @brief Convert optimizerAlgo enum to string representation */
std::string algoTypeToString(optimizerAlgo algo);

/** @brief Convert schedulerFunc enum to string representation */
std::string schedulerTypeToString(schedulerFunc func);

// ============================================================================
// Ensemble Memory Management Functions
// ============================================================================

/**
 * @brief Allocate ensemble metadata structure
 * 
 * Creates and initializes metadata with the specified configuration parameters.
 * 
 * @param max_trees Maximum number of trees to allocate space for
 * @param max_leaves Maximum number of leaves to allocate space for
 * @param max_trees_batch Maximum trees to add in a single batch
 * @param max_leaves_batch Maximum leaves to add in a single batch
 * @param input_dim Number of input features
 * @param output_dim Dimension of output space
 * @param policy_dim Dimension of policy (for RL applications)
 * @param max_depth Maximum tree depth
 * @param min_data_in_leaf Minimum samples required per leaf
 * @param n_bins Number of candidate bins for splits
 * @param par_th Parallelization threshold
 * @param cv_beta Control variates coefficient
 * @param verbose Verbosity level
 * @param batch_size Training batch size
 * @param use_cv Whether to use control variates
 * @param split_score_func Function for scoring splits
 * @param generator_type Method for generating split candidates
 * @param grow_policy Tree growing strategy
 * @return Pointer to allocated ensembleMetaData structure
 */
ensembleMetaData* ensemble_metadata_alloc(
    int max_trees, int max_leaves, int max_trees_batch, int max_leaves_batch,
    int input_dim, int output_dim, int policy_dim, int max_depth,
    int min_data_in_leaf, int n_bins, int par_th, float cv_beta,
    int verbose, int batch_size, bool use_cv, scoreFunc split_score_func,
    generatorType generator_type, growPolicy grow_policy
);

/**
 * @brief Allocate ensemble data storage
 * 
 * @param metadata Ensemble metadata specifying structure
 * @return Pointer to allocated ensembleData structure
 */
ensembleData* ensemble_data_alloc(ensembleMetaData *metadata);

/**
 * @brief Allocate a copy of ensemble data structure
 * 
 * @param metadata Ensemble metadata specifying structure
 * @return Pointer to newly allocated ensembleData copy
 */
ensembleData* ensemble_copy_data_alloc(ensembleMetaData *metadata);

/**
 * @brief Deep copy ensemble data
 * 
 * @param other_edata Source ensemble data to copy from
 * @param metadata Metadata describing the structure
 * @return Pointer to new ensembleData with copied values
 */
ensembleData* copy_ensemble_data(
    ensembleData *other_edata,
    ensembleMetaData *metadata
);

/**
 * @brief Deallocate ensemble data and free memory
 * 
 * @param edata Ensemble data to deallocate
 */
void ensemble_data_dealloc(ensembleData *edata);

/**
 * @brief Save ensemble data to binary file
 * 
 * @param file Output file stream
 * @param edata Ensemble data to save
 * @param metadata Ensemble metadata
 * @param device Device where data currently resides
 */
void save_ensemble_data(
    std::ofstream& file,
    ensembleData *edata,
    ensembleMetaData *metadata,
    deviceType device
);

/**
 * @brief Export ensemble model to header file format
 * 
 * Exports the model in a format suitable for deployment in embedded
 * systems or inference-only applications.
 * 
 * @param header_file Output header file stream
 * @param model_name Name of the model for code generation
 * @param edata Ensemble data to export
 * @param metadata Ensemble metadata
 * @param device Device where data currently resides
 * @param opts Vector of optimizer configurations
 * @param export_format Numerical format (float, fixed-point, etc.)
 * @param export_type Export style (full or compact)
 * @param prefix String prefix for generated code symbols
 */
void export_ensemble_data(
    std::ofstream& header_file,
    const std::string& model_name,
    ensembleData *edata,
    ensembleMetaData *metadata,
    deviceType device,
    std::vector<Optimizer*> opts,
    exportFormat export_format,
    exportType export_type,
    const std::string &prefix
);

/**
 * @brief Load ensemble data from binary file
 * 
 * @param file Input file stream
 * @param metadata Ensemble metadata (must match saved model)
 * @return Pointer to loaded ensembleData
 */
ensembleData* load_ensemble_data(
    std::ifstream& file,
    ensembleMetaData *metadata
);

/**
 * @brief Allocate memory for ensemble data arrays
 * 
 * @param metadata Ensemble metadata specifying sizes
 * @param edata Ensemble data structure to allocate arrays for
 */
void allocate_ensemble_memory(
    ensembleMetaData *metadata,
    ensembleData *edata
);

#endif // TYPES_H 