//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file gbrl.h
 * @brief Main GBRL (Gradient Boosted Reinforcement Learning) class
 * 
 * Provides the primary interface for training and using gradient boosted
 * tree ensembles. Supports both CPU and GPU execution, model serialization,
 * SHAP value computation, and various training configurations.
 */

#ifndef GBRL_H
#define GBRL_H

#include <string>
#include <tuple>

#include "node.h"
#include "optimizer.h"
#include "loss.h"
#include "split_candidate_generator.h"
#include "types.h"

#ifdef USE_CUDA
#include "cuda_types.h"
#endif

/**
 * @brief Main class for Gradient Boosted Reinforcement Learning
 * 
 * GBRL implements gradient boosted decision tree ensembles with support
 * for multi-output regression, custom optimization strategies, and both
 * CPU and GPU acceleration. Designed for reinforcement learning applications
 * but applicable to general supervised learning tasks.
 */
class GBRL {
    public:
        /**
         * @brief Construct GBRL model with enum parameters
         * 
         * @param input_dim Number of numerical input features
         * @param output_dim Dimensionality of output space
         * @param policy_dim Dimensionality of policy (for RL applications)
         * @param max_depth Maximum depth of individual trees
         * @param min_data_in_leaf Minimum samples required in each leaf
         * @param n_bins Number of candidate bins for splits
         * @param par_th Parallelization threshold
         * @param cv_beta Control variates beta parameter
         * @param split_score_func Scoring function for splits (L2/Cosine)
         * @param generator_type Candidate generation method (Uniform/Quantile)
         * @param use_control_variates Whether to use control variates
         * @param batch_size Training batch size
         * @param grow_policy Tree growing strategy (Greedy/Oblivious)
         * @param verbose Verbosity level
         * @param _device Device for computation (CPU/GPU)
         */
        GBRL(
            int input_dim, int output_dim, int policy_dim,
            int max_depth, int min_data_in_leaf,
            int n_bins, int par_th, float cv_beta,
            scoreFunc split_score_func,
            generatorType generator_type,
            bool use_control_variates,
            int batch_size,
            growPolicy grow_policy,
            int verbose,
            deviceType _device
        );
        
        /**
         * @brief Construct GBRL model with string parameters
         * 
         * @param input_dim Number of numerical input features
         * @param output_dim Dimensionality of output space
         * @param policy_dim Dimensionality of policy (for RL applications)
         * @param max_depth Maximum depth of individual trees
         * @param min_data_in_leaf Minimum samples required in each leaf
         * @param n_bins Number of candidate bins for splits
         * @param par_th Parallelization threshold
         * @param cv_beta Control variates beta parameter
         * @param split_score_func Scoring function name ("L2"/"Cosine")
         * @param generator_type Candidate generation method ("Uniform"/"Quantile")
         * @param use_control_variates Whether to use control variates
         * @param batch_size Training batch size
         * @param grow_policy Tree growing strategy ("Greedy"/"Oblivious")
         * @param verbose Verbosity level
         * @param _device Device for computation ("cpu"/"gpu")
         */
        GBRL(
            int input_dim, int output_dim, int policy_dim,
            int max_depth, int min_data_in_leaf,
            int n_bins, int par_th, float cv_beta,
            std::string split_score_func,
            std::string generator_type,
            bool use_control_variates,
            int batch_size,
            std::string grow_policy,
            int verbose,
            std::string _device
        );
        
        /**
         * @brief Construct GBRL model by loading from file
         * 
         * @param filename Path to saved model file
         */
        GBRL(const std::string& filename);
        
        /**
         * @brief Copy constructor
         * 
         * @param other GBRL model to copy from
         */
        GBRL(GBRL& other);
        
        /**
         * @brief Destructor - frees all allocated resources
         */
        ~GBRL();
        
        /**
         * @brief Compute SHAP values for a single tree
         * 
         * @param tree_idx Index of tree to explain
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param n_samples Number of samples
         * @param norm Normalization values
         * @param base_poly Base polynomial coefficients
         * @param offset Offset polynomial coefficients
         * @return Pointer to SHAP values array, caller must free
         */
        float* tree_shap(
            const int tree_idx,
            const float *obs,
            const char *categorical_obs,
            const int n_samples,
            float *norm,
            float *base_poly,
            float *offset
        );
        
        /**
         * @brief Compute SHAP values for entire ensemble
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param n_samples Number of samples
         * @param norm Normalization values
         * @param base_poly Base polynomial coefficients
         * @param offset Offset polynomial coefficients
         * @return Pointer to SHAP values array, caller must free
         */
        float* ensemble_shap(
            const float *obs,
            const char *categorical_obs,
            const int n_samples,
            float *norm,
            float *base_poly,
            float *offset
        );
        
        /**
         * @brief Check if CUDA is available
         * 
         * @return true if CUDA support is compiled and device is available
         */
        static bool cuda_available();
        
        /**
         * @brief Move model to specified device
         * 
         * @param _device Target device (CPU/GPU)
         */
        void to_device(deviceType _device);
        
        /**
         * @brief Get current device as string
         * 
         * @return Device name ("cpu" or "gpu")
         */
        std::string get_device();
        
        /**
         * @brief Save model to binary file
         * 
         * @param filename Output file path
         * @return 0 on success, error code otherwise
         */
        int saveToFile(const std::string& filename);
        
        /**
         * @brief Export model to C header file for deployment
         * 
         * @param filename Output header file path
         * @param modelname Name for the generated model
         * @param export_format Numerical format ("float"/"fxp8"/"fxp16")
         * @param export_type Export style ("full"/"compact")
         * @param prefix String prefix for generated symbols
         * @return 0 on success, error code otherwise
         */
        int exportModel(
            const std::string& filename,
            const std::string& modelname,
            const std::string& export_format,
            const std::string &export_type,
            const std::string& prefix
        );
        
        /**
         * @brief Load model from binary file
         * 
         * @param filename Input file path
         * @return 0 on success, error code otherwise
         */
        int loadFromFile(const std::string& filename);
        
        /**
         * @brief Validate ensemble consistency
         * 
         * Performs sanity checks on ensemble structure.
         */
        void ensemble_check();
        /**
         * @brief Perform single gradient boosting step
         * 
         * Adds one tree to ensemble by fitting to provided gradients.
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param grads Gradient values to fit
         * @param n_samples Number of samples
         * @param n_num_features Number of numerical features
         * @param n_cat_features Number of categorical features
         */
        void step(
            dataHolder<const float> *obs,
            dataHolder<const char> *categorical_obs,
            dataHolder<float> *grads,
            const int n_samples,
            const int n_num_features,
            const int n_cat_features
        );

        /**
         * @brief Generate predictions from ensemble
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param n_samples Number of samples
         * @param n_num_features Number of numerical features
         * @param n_cat_features Number of categorical features
         * @param start_tree_idx Starting tree index (inclusive)
         * @param stop_tree_idx Stopping tree index (exclusive)
         * @return Pointer to predictions (n_samples x output_dim), caller must free
         */
        float* predict(
            dataHolder<const float> *obs,
            dataHolder<const char> *categorical_obs,
            const int n_samples,
            const int n_num_features,
            const int n_cat_features,
            int start_tree_idx,
            int stop_tree_idx
        );

        /**
         * @brief Train ensemble on dataset
         * 
         * Main training loop that grows ensemble for specified iterations.
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param targets Target values (n_samples x output_dim)
         * @param iterations Number of boosting iterations
         * @param n_samples Number of samples
         * @param n_num_features Number of numerical features
         * @param n_cat_features Number of categorical features
         * @param shuffle Whether to shuffle data each iteration
         * @param _loss_type Loss function name (default: "MultiRMSE")
         * @return Final loss value
         */
        float fit(
            dataHolder<float> *obs,
            dataHolder<char> *categorical_obs,
            dataHolder<float> *targets,
            int iterations,
            const int n_samples,
            const int n_num_features,
            const int n_cat_features,
            bool shuffle = true,
            std::string _loss_type = "MultiRMSE"
        );
        
#ifdef USE_CUDA
        /**
         * @brief Perform single boosting step on GPU (internal)
         * 
         * @param dataset Training dataset on GPU
         */
        void _step_gpu(dataSet *dataset);
        
        /**
         * @brief Train ensemble on GPU (internal)
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param targets Target values
         * @param indices Sample indices
         * @param n_iterations Number of iterations
         * @param n_samples Number of samples
         * @param shuffle Whether to shuffle data
         * @return Final loss value
         */
        float _fit_gpu(
            dataHolder<float> *obs,
            dataHolder<char> *categorical_obs,
            dataHolder<float> *targets,
            std::vector<int> indices,
            const int n_iterations,
            const int n_samples,
            bool shuffle
        );
#endif

        /**
         * @brief Set global bias term
         * 
         * Sets the global bias/intercept term for the ensemble output. Supports both CPU
         * and GPU data through the dataHolder interface, with automatic device-to-device
         * transfers.
         * 
         * @param bias Bias values (output_dim) wrapped in dataHolder with device info
         * @param output_dim Output dimensionality (must match metadata->output_dim)
         * @throws std::runtime_error if output_dim doesn't match metadata->output_dim
         */
        void set_bias(dataHolder<const float> *bias, const int output_dim);
        
        /**
         * @brief Set per-feature importance weights
         * 
         * Sets the importance weight for each input feature. These weights are used during
         * split scoring to scale the contribution of each feature. Supports both CPU and GPU
         * data through the dataHolder interface, with automatic device-to-device transfers.
         * 
         * @param feature_weights Feature weights (input_dim) wrapped in dataHolder with device info
         * @param input_dim Input dimensionality (must match metadata->input_dim)
         * @throws std::runtime_error if input_dim doesn't match metadata->input_dim
         */
        void set_feature_weights(dataHolder<float> *feature_weights, const int input_dim);
        
        /**
         * @brief Set feature mapping for mixed categorical/numerical inputs
         * 
         * Configures how original feature indices map to internal feature representations,
         * enabling proper handling of datasets with both numerical and categorical features.
         * Creates 4 internal arrays:
         * - feature_mapping: Maps original feature indices to internal indices (stored for export)
         * - mapping_numerics: Flags indicating numerical (true) vs categorical (false) features (stored for export)
         * - reverse_num_feature_mapping: Maps internal numerical feature indices back to original (used in computation)
         * - reverse_cat_feature_mapping: Maps internal categorical feature indices back to original (used in computation)
         * 
         * Only the two reverse mapping arrays are actively used during training/prediction for
         * looking up feature weights. The forward mapping arrays are maintained for model
         * serialization and documentation purposes.
         * 
         * @param feature_mapping Array mapping original feature indices to internal indices (length: input_dim)
         * @param mapping_numerics Boolean array indicating if each original feature is numerical (length: input_dim)
         * @param input_dim Total number of input features (must match metadata->input_dim)
         */
        void set_feature_mapping(const int *feature_mapping, const bool *mapping_numerics, const int input_dim);
        
        /**
         * @brief Get current bias term
         * 
         * @return Pointer to bias array
         */
        float* get_bias();
        
        /**
         * @brief Get current feature weights
         * 
         * @return Pointer to feature weights array
         */
        float* get_feature_weights();
        
        /**
         * @brief Get feature mapping arrays
         * 
         * Returns copies of feature_mapping and mapping_numerics arrays.
         * Caller must delete[] both returned pointers.
         * 
         * @param feature_mapping Output pointer to feature mapping array (int*)
         * @param mapping_numerics Output pointer to numerics mask array (bool*)
         */
        void get_feature_mapping(int*& feature_mapping, bool*& mapping_numerics);
        
        /**
         * @brief Get learning rates from all schedulers
         * 
         * @return Pointer to learning rates array
         */
        float* get_scheduler_lrs();

        /**
         * @brief Get number of trees in ensemble
         * 
         * @return Current tree count
         */
        int get_num_trees();
        
        /**
         * @brief Get current training iteration
         * 
         * @return Iteration number
         */
        int get_iteration();

        /**
         * @brief Configure optimizer for tree range
         * 
         * @param algo Optimizer algorithm (SGD/Adam)
         * @param scheduler_func Learning rate scheduler (Const/Linear)
         * @param init_lr Initial learning rate
         * @param start_idx Starting tree index for this optimizer
         * @param stop_idx Stopping tree index for this optimizer
         * @param stop_lr Final learning rate (for Linear scheduler)
         * @param T Total iterations (for Linear scheduler)
         * @param beta_1 Adam beta_1 parameter
         * @param beta_2 Adam beta_2 parameter
         * @param eps Adam epsilon parameter
         * @param shrinkage Shrinkage/learning rate multiplier
         */
        void set_optimizer(
            optimizerAlgo algo,
            schedulerFunc scheduler_func,
            float init_lr,
            int start_idx,
            int stop_idx,
            float stop_lr,
            int T,
            float beta_1,
            float beta_2,
            float eps,
            float shrinkage
        );

        /**
         * @brief Print tree structure to console
         * 
         * @param tree_idx Index of tree to print
         */
        void print_tree(int tree_idx);
        
        /**
         * @brief Print ensemble metadata to console
         */
        void print_ensemble_metadata();
        
        /**
         * @brief Export tree visualization to DOT file
         * 
         * @param tree_idx Index of tree to plot
         * @param filename Output DOT file path
         */
        void plot_tree(int tree_idx, const std::string &filename);

        /**
         * @brief Get pointer to ensemble data structure
         * 
         * @return Pointer to ensembleData
         */
        ensembleData* get_ensemble_data();

        ensembleData *edata;                /**< Ensemble parameter data */
        ensembleMetaData *metadata;         /**< Ensemble metadata */
        serializationHeader sheader;        /**< Serialization header */
        std::vector<Optimizer*> opts;       /**< Optimizers for leaf updates */
        deviceType device = unspecified;    /**< Current compute device */
        bool parallel_predict = true;       /**< Enable parallel prediction */
        
#ifdef USE_CUDA
        SGDOptimizerGPU** cuda_opt = nullptr;  /**< GPU optimizers */
        int n_cuda_opts;                       /**< Number of GPU optimizers */
#endif 
};

#ifdef USE_CUDA
/**
 * @brief Validate CUDA device availability
 * 
 * @return true if valid CUDA device is available
 */
bool valid_device();
#endif

#endif // GBRL_H 