//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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
 * @file binding.cpp
 * @brief Python bindings for GBRL using pybind11
 * 
 * Provides Python interface to the C++ GBRL implementation, handling
 * NumPy arrays, PyTorch tensors, and DLPack for efficient data interchange.
 */

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#ifdef USE_CUDA
#include <cuda_runtime.h>  // For cudaMalloc, cudaFree
#endif

#include "gbrl.h"
#include "types.h"
#include "ensemble_io.h"
#include "dlpack/dlpack.h"

namespace py = pybind11;

/**
 * @brief Extract NumPy array information
 * 
 * @tparam T Array element type
 * @param obj Python object (should be NumPy array)
 * @param ptr Output pointer to array data
 * @param shape Output array shape
 * @param expected_format Expected data format string (optional)
 * @throws std::runtime_error if object is not a valid NumPy array
 */
template <typename T>
void get_numpy_array_info(
    py::object obj,
    T*& ptr,
    std::vector<size_t>& shape,
    const std::string& expected_format = ""
) {
    // Check if the object is a NumPy array
    if (!py::isinstance<py::array>(obj)) {
        throw std::runtime_error("Expected a NumPy array");
    }
    
    py::array arr = py::array::ensure(obj, py::array::c_style | py::array::forcecast);
    if (!arr) {
        throw std::runtime_error("Could not convert object to a contiguous NumPy array");
    }
    
    py::buffer_info info = arr.request();
    
    // Determine the expected format
    std::string expected;
    if (expected_format.empty()) {
        expected = py::format_descriptor<std::remove_cv_t<T>>::format();
    } else {
        expected = expected_format;
    }
    
    // Verify the data format
    if (info.format != expected) {
        std::stringstream ss;
        ss << "Expected array of format '" << expected << "', but got '" << info.format << "'";
        throw std::runtime_error(ss.str());
    }
    
    // Extract the data pointer, shape, and item size
    ptr = static_cast<T*>(info.ptr);
    shape.assign(info.shape.begin(), info.shape.end());
}

/**
 * @brief Extract tensor information from PyTorch tensor tuple
 * 
 * @tparam T Tensor element type
 * @param tensor_info Tuple containing (data_ptr, shape, dtype, device)
 * @param ptr Output pointer to tensor data
 * @param shape Output tensor shape
 * @param device Output device string ("cpu" or "cuda")
 * @throws std::runtime_error if tuple format is invalid
 */
template <typename T>
void get_tensor_info(
    py::tuple tensor_info,
    T*& ptr,
    std::vector<size_t>& shape,
    std::string& device
) {
    if (tensor_info.size() != 4) {
        throw std::runtime_error("Expected a tuple of size 4: (data_ptr, shape, dtype, device)");
    }
    
    size_t raw_ptr = tensor_info[0].cast<uintptr_t>();

    if (raw_ptr == 0 || raw_ptr == (size_t) - 1) {  // Check for null or invalid pointer values
        std::cerr << "ERROR: Extracted an invalid pointer! Setting ptr to nullptr." << std::endl;
        ptr = nullptr;
    } else {
        ptr = reinterpret_cast<T*>(raw_ptr);
    }

    if (ptr) {
        if (reinterpret_cast<uintptr_t>(ptr) % alignof(T) != 0) {
            std::cerr << "ERROR: Pointer is not properly aligned! Possible misaligned memory access." << std::endl;
        }
    }
    
    // Extract shape
    py::tuple shape_tuple = tensor_info[1].cast<py::tuple>();
    shape.clear();
    for (py::handle dim : shape_tuple) {
        shape.push_back(dim.cast<size_t>());
    }
    
    // Extract and verify dtype
    std::string dtype = tensor_info[2].cast<std::string>();
    std::string expected_dtype;
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        expected_dtype = "torch.float32";
    } else if (std::is_same<T, double>::value || std::is_same<T, const double>::value) {
        expected_dtype = "torch.float64";
    } else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        expected_dtype = "torch.int32";
    } else {
        throw std::runtime_error("Unsupported data type: " + dtype);
    }
    
    if (dtype != expected_dtype) {
        throw std::runtime_error("Expected dtype " + expected_dtype + ", but got " + dtype);
    }
    
    // Extract device
    device = tensor_info[3].cast<std::string>();
}

/**
 * @brief Handle input from Python (NumPy array or PyTorch tensor)
 * 
 * @tparam T Element type
 * @param input Python input object
 * @param ptr Output data pointer
 * @param shape Output shape vector
 * @param device Output device string
 * @param name Parameter name for error messages
 * @param none_allowed Whether None input is allowed
 * @param function_name Function name for error messages
 * @param expected_format Expected data format (optional)
 * @throws std::runtime_error if input is invalid
 */
template <typename T>
void handle_input_info(
    py::object& input,
    T*& ptr,
    std::vector<size_t>& shape,
    std::string& device,
    const std::string& name,
    const bool none_allowed,
    const std::string& function_name,
    const std::string& expected_format = ""
) {
    if (input.is_none()) {
        if (!none_allowed) {
            throw std::runtime_error("Cannot call " + function_name + " without " + name + "!");
        } else {
            device = "cpu";
            ptr = nullptr;
            return;
        }
    }
    
    if (py::isinstance<py::array>(input)) {
        get_numpy_array_info<T>(input, ptr, shape, expected_format);
        device = "cpu";
    } else if (py::isinstance<py::tuple>(input)) {
        get_tensor_info<T>(input, ptr, shape, device);
    } else {
        throw std::runtime_error("Unknown " + name + " type! Must be a NumPy array or tuple.");
    }
}

/**
 * @brief DLPack tensor deleter function
 * 
 * Frees memory for DLPack managed tensors on CPU or GPU.
 * 
 * @param self DLManagedTensor to delete
 */
void dlpack_deleter_function(DLManagedTensor* self) {
#ifdef USE_CUDA
    if (self->dl_tensor.device.device_type == kDLCUDA) {
        cudaFree(static_cast<float*>(self->dl_tensor.data));
    }
#endif 
    if (self->dl_tensor.device.device_type == kDLCPU) {
        delete[] static_cast<float*>(self->dl_tensor.data);
    }
    delete[] self->dl_tensor.shape;
    delete self;
}

/**
 * @brief Create DLPack tensor capsule for Python interop
 * 
 * @param raw_ptr Pointer to tensor data
 * @param shape Tensor shape
 * @param dtype Data type descriptor
 * @param device Device descriptor
 * @return PyCapsule containing DLManagedTensor
 */
py::capsule create_dlpack_tensor(
    void* raw_ptr,
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DLDevice device
) {
    // Allocate memory for DLTensor
    DLManagedTensor* managed_tensor = new DLManagedTensor;
    
    // Assign raw pointer and device
    managed_tensor->dl_tensor.data = raw_ptr;
    managed_tensor->dl_tensor.device = device;
    
    // Set shape
    managed_tensor->dl_tensor.ndim = static_cast<int32_t>(shape.size());
    managed_tensor->dl_tensor.shape = new int64_t[shape.size()];
    std::copy(shape.begin(), shape.end(), managed_tensor->dl_tensor.shape);
    
    // Set dtype
    managed_tensor->dl_tensor.dtype = dtype;
    
    // Strides are optional; set to nullptr for default behavior
    managed_tensor->dl_tensor.strides = nullptr;
    managed_tensor->dl_tensor.byte_offset = 0;

    // Set the deleter function
    managed_tensor->manager_ctx = nullptr;
    // Set the deleter function
    managed_tensor->deleter = dlpack_deleter_function;
    
    // Return the PyCapsule containing the DLManagedTensor
    return py::capsule(managed_tensor, "dltensor");
}

/**
 * @brief Return tensor information as NumPy array or DLPack capsule
 * 
 * @param num_samples Number of samples
 * @param output_dim Output dimensionality
 * @param ptr Data pointer
 * @param device Device type
 * @param is_torch Whether to return as PyTorch-compatible DLPack
 * @return Python object (NumPy array or DLPack capsule)
 */
py::object return_tensor_info(
    int num_samples,
    int output_dim,
    float *ptr,
    deviceType device,
    bool is_torch
) {
    // Allocate memory
    std::vector<int64_t> shape;
    if (output_dim == 1) {
        shape = {static_cast<int64_t>(num_samples)};  // 1D case
    } else {
        shape = {static_cast<int64_t>(num_samples), static_cast<int64_t>(output_dim)};  // 2D case
    }
#ifdef USE_CUDA
    if (device == gpu && !is_torch)
        is_torch = true;
#endif
    if (is_torch){
        DLDevice device_dl = {kDLCPU, 0};
        DLDataType dtype = {kDLFloat, 32, 1};
#ifdef USE_CUDA
    if (device == deviceType::gpu){
        device_dl.device_type = kDLCUDA;
    }
#endif
        return create_dlpack_tensor(static_cast<void*>(ptr), shape, dtype, device_dl);
    }
    else {
    auto capsule = py::capsule(ptr, [](void* p) {
            delete []reinterpret_cast<float*>(p);});
            // Return a NumPy array for CPU case
        return py::array_t<float>(shape, ptr, capsule);
    }
}

/**
 * @brief Convert ensemble metadata to a Python dictionary
 * 
 * @param metadata Pointer to ensembleMetaData (may be nullptr, returns empty dict)
 * @return Python dict with all hyperparameters and configuration values
 */
py::dict metadataToDict(const ensembleMetaData* metadata){
    py::dict d;
    if (metadata != nullptr){
        d["input_dim"] = metadata->input_dim;
        d["output_dim"] = metadata->output_dim;
        d["policy_dim"] = metadata->policy_dim;
        d["split_score_func"] = scoreFuncToString(metadata->split_score_func);
        d["generator_type"] = generatorTypeToString(metadata->generator_type);
        d["use_control_variates"] = metadata->use_cv;
        d["verbose"] = metadata->verbose;
        d["max_depth"] = metadata->max_depth;
        d["min_data_in_leaf"] = metadata->min_data_in_leaf;
        d["n_bins"] = metadata->n_bins;
        d["par_th"] = metadata->par_th;
        d["batch_size"] = metadata->batch_size;
        d["grow_policy"] = growPolicyToString(metadata->grow_policy);
        d["n_objs"] = metadata->n_objs;
        d["lambda_penalty"] = metadata->lambda_penalty;
        d["iteration"] = metadata->iteration;
    }
    return d;
}

/**
 * @brief Convert full ensemble data to a Python dictionary of NumPy arrays
 * 
 * Transfers ownership of all dynamically allocated arrays in ensembleData
 * (and its sub-structs) to NumPy via pybind11 capsules. After calling this
 * function, the arrays are managed by Python's garbage collector.
 * 
 * @param data Pointer to ensembleData (may be nullptr, returns empty dict)
 * @param metadata Pointer to ensembleMetaData describing array dimensions
 * @return Python dict with tree structure, leaf values, feature data, and mappings
 */
py::dict ensembleDataToDict(const ensembleData* data, const ensembleMetaData* metadata) {
    py::dict d;
    if (data != nullptr) {
        // Convert float pointers to NumPy arrays with ownership transfer
        auto bias_capsule = py::capsule(data->bias, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["bias"] = py::array_t<float>({metadata->output_dim}, data->bias, bias_capsule);

        auto feature_mapping_capsule = py::capsule(data->feature_mappings->feature_mapping, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["feature_mapping"] = py::array_t<int>({metadata->input_dim}, data->feature_mappings->feature_mapping, feature_mapping_capsule);
        
        auto reverse_num_feature_mapping_capsule = py::capsule(data->feature_mappings->reverse_num_feature_mapping, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["reverse_num_feature_mapping"] = py::array_t<int>({metadata->input_dim}, data->feature_mappings->reverse_num_feature_mapping, reverse_num_feature_mapping_capsule);
        
        auto reverse_cat_feature_mapping_capsule = py::capsule(data->feature_mappings->reverse_cat_feature_mapping, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["reverse_cat_feature_mapping"] = py::array_t<int>({metadata->input_dim}, data->feature_mappings->reverse_cat_feature_mapping, reverse_cat_feature_mapping_capsule);
        
        auto feature_weights_capsule = py::capsule(data->feature_data->feature_weights, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["feature_weights"] = py::array_t<float>({metadata->input_dim}, data->feature_data->feature_weights, feature_weights_capsule);

        auto tree_indices_capsule = py::capsule(data->ensemble_info->tree_indices, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["tree_indices"] = py::array_t<int>({metadata->n_trees}, data->ensemble_info->tree_indices, tree_indices_capsule);

        int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;

        auto depths_capsule = py::capsule(data->ensemble_info->depths, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["depths"] = py::array_t<int>({split_sizes}, data->ensemble_info->depths, depths_capsule);

        auto values_capsule = py::capsule(data->leaf_data->values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["values"] = py::array_t<float>({metadata->n_leaves, metadata->output_dim}, data->leaf_data->values, values_capsule);

        auto feature_indices_capsule = py::capsule(data->feature_data->feature_indices, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["feature_indices"] = py::array_t<int>({split_sizes, metadata->max_depth}, data->feature_data->feature_indices, feature_indices_capsule);

        auto feature_values_capsule = py::capsule(data->feature_data->feature_values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["feature_values"] = py::array_t<float>({split_sizes, metadata->max_depth}, data->feature_data->feature_values, feature_values_capsule);

        auto edge_weights_capsule = py::capsule(data->leaf_data->edge_weights, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["edge_weights"] = py::array_t<float>({metadata->n_leaves, metadata->max_depth}, data->leaf_data->edge_weights, edge_weights_capsule);

        auto is_numerics_capsule = py::capsule(data->feature_data->is_numerics, [](void* ptr) { delete[] reinterpret_cast<bool*>(ptr); });
        d["is_numerics"] = py::array_t<bool>({split_sizes, metadata->max_depth}, data->feature_data->is_numerics, is_numerics_capsule);

        auto inequality_directions_capsule = py::capsule(data->feature_data->inequality_directions, [](void* ptr) { delete[] reinterpret_cast<bool*>(ptr); });
        d["inequality_directions"] = py::array_t<bool>({metadata->n_leaves, metadata->max_depth}, data->feature_data->inequality_directions, inequality_directions_capsule);
        
        auto mapping_numerics_capsule = py::capsule(data->feature_mappings->mapping_numerics, [](void* ptr) { delete[] reinterpret_cast<bool*>(ptr); });
        d["mapping_numerics"] = py::array_t<bool>({metadata->input_dim}, data->feature_mappings->mapping_numerics, mapping_numerics_capsule);

        // Convert char* categorical_values to NumPy string array (S128)
        auto categorical_capsule = py::capsule(data->feature_data->categorical_values, [](void* ptr) { delete[] reinterpret_cast<char*>(ptr); });
        d["categorical_values"] = py::array(py::dtype("S128"), {split_sizes, metadata->max_depth}, data->feature_data->categorical_values, categorical_capsule);
        
        d["alloc_data_size"] = data->alloc_data_size;

#ifdef DEBUG
        auto n_samples_capsule = py::capsule(data->n_samples, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["n_samples"] = py::array_t<int>({metadata->n_leaves}, data->n_samples, n_samples_capsule);
#endif
    }
    return d;
}

/**
 * @brief Convert exportData struct to a Python dictionary of NumPy arrays
 * 
 * Transfers ownership of all dynamically allocated arrays in exportData
 * to NumPy via pybind11 capsules. After calling this function, the arrays
 * are managed by Python's garbage collector. The caller must still delete
 * the exportData struct itself (but NOT its array members).
 * 
 * @param data Pointer to exportData (may be nullptr, returns empty dict)
 * @return Python dict with keys: n_trees, n_leaves, input_dim, output_dim,
 *         max_depth, num_features, binary_features, bias, feature_indices,
 *         feature_values, leaf_values
 */
py::dict ensembleExportDataToDict(const exportData* data) {
    py::dict d;
    if (data != nullptr) {
        // Scalar metadata
        d["n_trees"] = data->n_trees;
        d["n_leaves"] = data->n_leaves;
        d["input_dim"] = data->input_dim;
        d["output_dim"] = data->output_dim;
        d["max_depth"] = data->max_depth;
        d["num_features"] = data->num_features;
        d["binary_features"] = data->binary_features;

        // Array data with ownership transfer via capsules
        auto bias_capsule = py::capsule(data->bias, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["bias"] = py::array_t<float>({data->output_dim}, data->bias, bias_capsule);

        auto feature_indices_capsule = py::capsule(data->feature_indices, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["feature_indices"] = py::array_t<int>({data->binary_features}, data->feature_indices, feature_indices_capsule);

        auto feature_values_capsule = py::capsule(data->feature_values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["feature_values"] = py::array_t<float>({data->binary_features}, data->feature_values, feature_values_capsule);

        auto leaf_values_capsule = py::capsule(data->leaf_values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["leaf_values"] = py::array_t<float>({data->n_leaves, data->output_dim}, data->leaf_values, leaf_values_capsule);
    }
    return d;
}


/**
 * @brief Convert optimizer configuration to a Python dictionary
 * 
 * Extracts all optimizer parameters (algorithm, learning rate, scheduler,
 * Adam betas, etc.) into a dict. Deletes the optimizerConfig struct
 * after conversion since it was allocated by getConfig().
 * 
 * @param conf Pointer to optimizerConfig (may be nullptr, returns empty dict).
 *             Deleted after conversion.
 * @return Python dict with optimizer configuration parameters
 */
py::dict optimizerToDict(const optimizerConfig* conf){
    py::dict d;
    if (conf != nullptr){
        d["algo"] = conf->algo;
        d["init_lr"] = conf->init_lr;
        d["start_idx"] = conf->start_idx;
        d["stop_idx"] = conf->stop_idx;
        d["scheduler_func"] = conf->scheduler_func;
        d["stop_lr"] = conf->stop_lr;
        d["T"] = conf->T;
        d["beta_1"] = conf->beta_1;
        d["beta_2"] = conf->beta_2;
        d["eps]"] = conf->eps;
        delete conf;  // Delete the struct pointer if it's no longer neede
    }
    
    return d;
}

/**
 * @brief Convert all optimizer configurations to a Python list of dicts
 * 
 * @param opts Vector of Optimizer pointers to extract configurations from
 * @return Python list where each element is a dict of optimizer parameters
 */
py::list getOptimizerConfigs(const std::vector<Optimizer*>& opts) {
    py::list configs;
    for (auto& opt : opts) {
        optimizerConfig* conf = opt->getConfig();
        configs.append(optimizerToDict(conf));  // conf is deleted within optimizerConfigToDict
    }
    return configs;
}

/**
 * @brief Convert a treeData struct to a Python dictionary of NumPy arrays
 * 
 * Transfers ownership of all treeData arrays to NumPy via capsules.
 * After this call the treeData struct shell can be deleted, but not
 * the arrays (they are now owned by NumPy).
 * 
 * @param tdata Tree data to convert (must not be nullptr)
 * @return Python dict with scalar metadata and NumPy array values
 */
py::dict treeDataToDict(treeData* tdata) {
    py::dict d;
    if (tdata == nullptr) return d;

    d["n_leaves"] = tdata->n_leaves;
    d["tree_depth"] = tdata->tree_depth;
    d["output_dim"] = tdata->output_dim;
    d["max_depth"] = tdata->max_depth;
    d["n_objs"] = tdata->n_objs;
    d["split_count"] = tdata->split_count;
    d["is_oblivious"] = tdata->is_oblivious;

    int n_leaves = tdata->n_leaves;
    int max_depth = tdata->max_depth;
    int output_dim = tdata->output_dim;
    int split_count = tdata->split_count;
    int n_objs = tdata->n_objs;
    int depth_count = tdata->is_oblivious ? 1 : n_leaves;

    auto depths_cap = py::capsule(tdata->depths, [](void* p){ delete[] reinterpret_cast<int*>(p); });
    d["depths"] = py::array_t<int>({depth_count}, tdata->depths, depths_cap);

    auto values_cap = py::capsule(tdata->values, [](void* p){ delete[] reinterpret_cast<float*>(p); });
    d["values"] = py::array_t<float>({n_leaves, output_dim}, tdata->values, values_cap);

    auto ew_cap = py::capsule(tdata->edge_weights, [](void* p){ delete[] reinterpret_cast<float*>(p); });
    d["edge_weights"] = py::array_t<float>({n_leaves, max_depth}, tdata->edge_weights, ew_cap);

    auto fi_cap = py::capsule(tdata->feature_indices, [](void* p){ delete[] reinterpret_cast<int*>(p); });
    d["feature_indices"] = py::array_t<int>({split_count, max_depth}, tdata->feature_indices, fi_cap);

    auto fv_cap = py::capsule(tdata->feature_values, [](void* p){ delete[] reinterpret_cast<float*>(p); });
    d["feature_values"] = py::array_t<float>({split_count, max_depth}, tdata->feature_values, fv_cap);

    auto in_cap = py::capsule(tdata->is_numerics, [](void* p){ delete[] reinterpret_cast<bool*>(p); });
    d["is_numerics"] = py::array_t<bool>({split_count, max_depth}, tdata->is_numerics, in_cap);

    auto id_cap = py::capsule(tdata->inequality_directions, [](void* p){ delete[] reinterpret_cast<bool*>(p); });
    d["inequality_directions"] = py::array_t<bool>({n_leaves, max_depth}, tdata->inequality_directions, id_cap);

    auto cv_cap = py::capsule(tdata->categorical_values, [](void* p){ delete[] reinterpret_cast<char*>(p); });
    d["categorical_values"] = py::array_t<char>({split_count * max_depth * MAX_CHAR_SIZE}, tdata->categorical_values, cv_cap);

    auto dens_cap = py::capsule(tdata->densities, [](void* p){ delete[] reinterpret_cast<float*>(p); });
    d["densities"] = py::array_t<float>({n_leaves, n_objs}, tdata->densities, dens_cap);

    return d;
}

/**
 * @brief Convert a Python dictionary back to a treeData struct
 * 
 * Allocates a new treeData and copies all array data from the dict's
 * NumPy arrays. The caller owns all memory and must free via tree_data_dealloc().
 * 
 * @param d Python dict as returned by treeDataToDict or constructed by the user
 * @return Newly allocated treeData with copies of all arrays
 */
treeData* dictToTreeData(const py::dict& d) {
    treeData *tdata = new treeData;

    tdata->n_leaves = d["n_leaves"].cast<int>();
    tdata->tree_depth = d["tree_depth"].cast<int>();
    tdata->output_dim = d["output_dim"].cast<int>();
    tdata->max_depth = d["max_depth"].cast<int>();
    tdata->n_objs = d["n_objs"].cast<int>();
    tdata->split_count = d["split_count"].cast<int>();
    tdata->is_oblivious = d["is_oblivious"].cast<bool>();

    int n_leaves = tdata->n_leaves;
    int max_depth = tdata->max_depth;
    int output_dim = tdata->output_dim;
    int split_count = tdata->split_count;
    int n_objs = tdata->n_objs;
    int depth_count = tdata->is_oblivious ? 1 : n_leaves;

    py::array_t<int> depths_arr = d["depths"].cast<py::array_t<int>>();
    tdata->depths = new int[depth_count];
    memcpy(tdata->depths, depths_arr.data(), depth_count * sizeof(int));

    py::array_t<float> values_arr = d["values"].cast<py::array_t<float>>();
    tdata->values = new float[n_leaves * output_dim];
    memcpy(tdata->values, values_arr.data(), n_leaves * output_dim * sizeof(float));

    py::array_t<float> ew_arr = d["edge_weights"].cast<py::array_t<float>>();
    tdata->edge_weights = new float[n_leaves * max_depth];
    memcpy(tdata->edge_weights, ew_arr.data(), n_leaves * max_depth * sizeof(float));

    py::array_t<int> fi_arr = d["feature_indices"].cast<py::array_t<int>>();
    tdata->feature_indices = new int[split_count * max_depth];
    memcpy(tdata->feature_indices, fi_arr.data(), split_count * max_depth * sizeof(int));

    py::array_t<float> fv_arr = d["feature_values"].cast<py::array_t<float>>();
    tdata->feature_values = new float[split_count * max_depth];
    memcpy(tdata->feature_values, fv_arr.data(), split_count * max_depth * sizeof(float));

    py::array_t<bool> in_arr = d["is_numerics"].cast<py::array_t<bool>>();
    tdata->is_numerics = new bool[split_count * max_depth];
    memcpy(tdata->is_numerics, in_arr.data(), split_count * max_depth * sizeof(bool));

    py::array_t<bool> id_arr = d["inequality_directions"].cast<py::array_t<bool>>();
    tdata->inequality_directions = new bool[n_leaves * max_depth];
    memcpy(tdata->inequality_directions, id_arr.data(), n_leaves * max_depth * sizeof(bool));

    py::array_t<char> cv_arr = d["categorical_values"].cast<py::array_t<char>>();
    tdata->categorical_values = new char[split_count * max_depth * MAX_CHAR_SIZE];
    memcpy(tdata->categorical_values, cv_arr.data(), split_count * max_depth * MAX_CHAR_SIZE * sizeof(char));

    py::array_t<float> dens_arr = d["densities"].cast<py::array_t<float>>();
    tdata->densities = new float[n_leaves * n_objs];
    memcpy(tdata->densities, dens_arr.data(), n_leaves * n_objs * sizeof(float));

    return tdata;
}

PYBIND11_MODULE(gbrl_cpp, m) {
    py::class_<GBRL> gbrl(m, "GBRL");
    gbrl.def(py::init<int, int, int, int, int, int, int, float, std::string, std::string, bool, int, std::string, int, int, float, std::string, std::string, int>(),
         py::arg("input_dim")=1, 
         py::arg("output_dim")=1, 
         py::arg("policy_dim")=1, 
         py::arg("max_depth")=4, 
         py::arg("min_data_in_leaf")=0, 
         py::arg("n_bins")=256, 
         py::arg("par_th")=10, 
         py::arg("cv_beta")=0.9,  
         py::arg("split_score_func")="cosine", 
         py::arg("generator_type")="quantile", 
         py::arg("use_control_variates")=false, 
         py::arg("batch_size")=5000, 
         py::arg("grow_policy")="greedy", 
         py::arg("n_objs")=1, 
         py::arg("verbose")=0,
         py::arg("lambda_penalty")=1.0,
         py::arg("device")="cpu",
         py::arg("learner_name")="GBRL",
         py::arg("n_mono_constraints")=0,
         "Constructor of the GBRL class");
    gbrl.def(py::init<GBRL&>(), py::arg("model"), "Copy constructor"); // This exposes the filename constructor
    gbrl.def_static("load", [](const std::string& filename) {
        return new GBRL(filename);  // Factory function creating a new instance
    }, py::return_value_policy::take_ownership,
    "Load a GBRL object from a file"); // Python takes ownership of the new instance
    // fit method
    gbrl.def("to_device", [](GBRL &self, std::string& str_device) {
        py::gil_scoped_release release; 
        self.to_device(stringTodeviceType(str_device)); 
    },  py::arg("device"),
    "Set GBRL device ['cpu', 'cuda']");
    gbrl.def("step", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::object &grads, py::object &obj_labels) {
        const float* obs_ptr = nullptr;
        const char* cat_obs_ptr= nullptr;
        float* grads_ptr = nullptr;
        const float* obj_labels_ptr = nullptr;
        std::vector<size_t> obs_shape, cat_obs_shape, grads_shape, obj_labels_shape;
        std::string obs_device, cat_obs_device, grads_device, obj_labels_device;
        int n_samples, n_num_features = 0, n_cat_features = 0, grad_output_dim, obj_labels_dim;
        int n_obs_samples, n_cat_samples, n_obj_labels_samples, n_objs;
        
        handle_input_info<float>(grads, grads_ptr, grads_shape, grads_device, "grads", false, "step");
        if (grads_shape.size() == 1){
            if (self.metadata->output_dim > 1){
                n_samples = 1;
                grad_output_dim = static_cast<int>(grads_shape[0]);
            } else{
                n_samples = static_cast<int>(grads_shape[0]);
                grad_output_dim = 1;
            }
            n_objs = 1;
        } else if (grads_shape.size() == 2){
            n_samples = static_cast<int>(grads_shape[0]);
            grad_output_dim = static_cast<int>(grads_shape[1]);
            n_objs = 1;
        } else {
            n_objs = static_cast<int>(grads_shape[0]);
            n_samples = static_cast<int>(grads_shape[1]);
            grad_output_dim = static_cast<int>(grads_shape[2]);
        }

        if (grad_output_dim != self.metadata->output_dim){
            std::stringstream ss;
            ss << "Gradient output dim " << grad_output_dim << " != correct output dim " << self.metadata->output_dim;
            throw std::runtime_error(ss.str());
        }

        if (n_objs != self.metadata->n_objs){
            std::stringstream ss;
            ss << "Number of objectives " << n_objs << " != correct number of objectives " << self.metadata->n_objs;
            throw std::runtime_error(ss.str());
        }

        dataHolder<float> grads_handler{grads_ptr, stringTodeviceType(grads_device)};

        handle_input_info<const float>(obs, obs_ptr, obs_shape, obs_device, "obs", true, "step");
        if (obs_ptr != nullptr){
            if (obs_shape.size() == 1){
               n_num_features = (n_samples == 1) ? static_cast<int>(obs_shape[0]) : 1;
               n_obs_samples = (n_samples == 1) ? 1 : static_cast<int>(obs_shape[0]);
            } else {
                n_obs_samples = static_cast<int>(obs_shape[0]);
                n_num_features = (obs_shape.size() == 1) ? 1 : static_cast<int>(obs_shape[1]);
            }
            if (n_obs_samples != n_samples){
                std::stringstream ss;
                ss << "Number of observations " << n_obs_samples << " != number of gradient samples " << n_samples;
                throw std::runtime_error(ss.str());
            }
        }

        dataHolder<const float> obs_handler{obs_ptr, stringTodeviceType(obs_device)};

        handle_input_info<const char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "step", CAT_TYPE);

        if (cat_obs_ptr != nullptr){
            if (cat_obs_shape.size() == 1){
               n_cat_features = (n_samples == 1) ? static_cast<int>(cat_obs_shape[0]) : 1;
               n_cat_samples = (n_samples == 1) ? 1 : static_cast<int>(cat_obs_shape[0]);
            } else {
                n_cat_samples = static_cast<int>(cat_obs_shape[0]);
                n_cat_features = (cat_obs_shape.size() == 1) ? 1 : static_cast<int>(cat_obs_shape[1]);
            }
            if (n_cat_samples != n_samples){
                std::stringstream ss;
                ss << "Number of categorical observations " << n_cat_samples << " != number of gradient samples " << n_samples;
                throw std::runtime_error(ss.str());
            }
        }

        if (n_cat_features + n_num_features != self.metadata->input_dim){
            std::stringstream ss;
            ss << "Total number of features " << n_cat_features + n_num_features << " != correct input dim " << self.metadata->input_dim;
            throw std::runtime_error(ss.str());
        }

        dataHolder<const char> cat_obs_handler{cat_obs_ptr, stringTodeviceType(cat_obs_device)};

        handle_input_info<const float>(obj_labels, obj_labels_ptr, obj_labels_shape, obj_labels_device, "obj_labels", true, "step");
        if (obj_labels_ptr != nullptr){
            if (obj_labels_shape.size() == 1){
                n_obj_labels_samples = static_cast<int>(obj_labels_shape[0]);
                if (n_obj_labels_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of obj_labels samples " << n_obj_labels_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            } else{
                if (obj_labels_shape.size() > 2){
                    std::stringstream ss;
                    ss << "obj_labels has invalid shape. Should be a vector";
                    throw std::runtime_error(ss.str());
                }
                n_obj_labels_samples = static_cast<int>(obj_labels_shape[0]);
                obj_labels_dim = static_cast<int>(obj_labels_shape[1]);

                if (n_obj_labels_samples == 1){
                    n_obj_labels_samples = obj_labels_dim;
                }

                if (n_obj_labels_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of obj_labels samples " << n_obj_labels_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            }
        }

        dataHolder<const float> obj_labels_handler{obj_labels_ptr, stringTodeviceType(obj_labels_device)};

        py::gil_scoped_release release;
        self.step(&obs_handler, &cat_obs_handler, &grads_handler, &obj_labels_handler, n_samples, n_num_features, n_cat_features);
    },  py::arg("obs"),
        py::arg("categorical_obs"),
        py::arg("grads"),
        py::arg("obj_labels")=py::none(),
    "Fit a decision tree with the given observations and gradients");
    gbrl.def("fit", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::object &targets, int iterations, bool shuffle, std::string loss_type) -> float {
        float* obs_ptr = nullptr;
        char* cat_obs_ptr = nullptr;
        float* targets_ptr = nullptr;
        std::vector<size_t> obs_shape, cat_obs_shape, targets_shape;
        std::string obs_device, cat_obs_device, targets_device;
        int n_samples, n_num_features = 0, n_cat_features = 0;
        int target_output_dim, n_obs_samples, n_cat_samples;

        handle_input_info<float>(targets, targets_ptr, targets_shape, targets_device, "targets", false, "fit");
        if (targets_shape.size() == 1){
            if (self.metadata->output_dim > 1){
                n_samples = 1;
                target_output_dim = static_cast<int>(targets_shape[0]);
            } else{
                n_samples = static_cast<int>(targets_shape[0]);
                target_output_dim = 1;
            }
        } else {
            n_samples = static_cast<int>(targets_shape[0]);
            target_output_dim = static_cast<int>(targets_shape[1]);
        }
        if (target_output_dim != self.metadata->output_dim){
                std::stringstream ss;
                ss << "Targets output dim " << target_output_dim << " != correct output dim " << self.metadata->output_dim;
                throw std::runtime_error(ss.str());
        }

        dataHolder<float> targets_handler{targets_ptr, stringTodeviceType(targets_device)};

        handle_input_info<float>(obs, obs_ptr, obs_shape, obs_device, "obs", true, "fit"); 
        if (obs_ptr != nullptr){
            if (obs_shape.size() == 1){
               n_num_features = (n_samples == 1) ? static_cast<int>(obs_shape[0]) : 1;
               n_obs_samples = (n_samples == 1) ? 1 : static_cast<int>(obs_shape[0]);
            } else {
                n_obs_samples = static_cast<int>(obs_shape[0]);
                n_num_features = (obs_shape.size() == 1) ? 1 : static_cast<int>(obs_shape[1]);
            }
                if (n_obs_samples != n_samples){
                std::stringstream ss;
                ss << "Number of observations " << n_obs_samples << " != number of gradient samples " << n_samples;
                throw std::runtime_error(ss.str());
            }
        }

        dataHolder<float> obs_handler{obs_ptr, stringTodeviceType(obs_device)};

        handle_input_info<char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "fit", CAT_TYPE);

        if (cat_obs_ptr != nullptr){
            if (cat_obs_shape.size() == 1){
               n_cat_features = (n_samples == 1) ? static_cast<int>(cat_obs_shape[0]) : 1;
               n_cat_samples = (n_samples == 1) ? 1 : static_cast<int>(cat_obs_shape[0]);
            } else {
                n_cat_samples = static_cast<int>(cat_obs_shape[0]);
                n_cat_features = (cat_obs_shape.size() == 1) ? 1 : static_cast<int>(cat_obs_shape[1]);
            }
            if (n_cat_samples != n_samples){
                std::stringstream ss;
                ss << "Number of categorical observations " << n_cat_samples << " != number of gradient samples " << n_samples;
                throw std::runtime_error(ss.str());
            }
        }


        dataHolder<char> cat_obs_handler{cat_obs_ptr, stringTodeviceType(cat_obs_device)};

        if (n_cat_features + n_num_features != self.metadata->input_dim){
            std::stringstream ss;
            ss << "Total number of features " << n_cat_features + n_num_features << " != correct input dim " << self.metadata->input_dim;
            throw std::runtime_error(ss.str());
        }

        py::gil_scoped_release release; 
        return self.fit(&obs_handler, &cat_obs_handler, &targets_handler, iterations, n_samples, n_num_features, n_cat_features, shuffle, loss_type); 
    },  py::arg("obs"),
        py::arg("categorical_obs"),
        py::arg("targets"),
        py::arg("iterations"),
        py::arg("shuffle")=true,  
        py::arg("loss_type")="MultiRMSE",  
    "Fit a decision tree with the given observations and targets for <iterations> boosting rounds");
    gbrl.def("set_bias", [](GBRL &self, py::object &bias) {
        const float *bias_ptr = nullptr;
        std::vector<size_t> bias_shape;
        std::string bias_device;
        int n_samples, bias_dim;

        handle_input_info<const float>(bias, bias_ptr, bias_shape, bias_device, "bias", false, "set_bias");

        if (bias_shape.size() == 1){
            if (self.metadata->output_dim > 1){
                n_samples = 1;
                bias_dim = static_cast<int>(bias_shape[0]);
            } else{
                n_samples = static_cast<int>(bias_shape[0]);
                bias_dim = 1;
            }

            if (n_samples > 1){
                std::stringstream ss;
                ss << "Set bias with multiple samples is not supported!";
                throw std::runtime_error(ss.str());
            }
        } else {
            n_samples = static_cast<int>(bias_shape[0]);
            bias_dim = static_cast<int>(bias_shape[1]);

            if (n_samples == self.metadata->output_dim && bias_dim == 1){
                // Transpose case
                n_samples = 1;
                bias_dim = static_cast<int>(bias_shape[0]);
            }
        }
        if (bias_dim != self.metadata->output_dim){
            std::stringstream ss;
            ss << "Targets output dim " << bias_dim << " != correct output dim " << self.metadata->output_dim;
            throw std::runtime_error(ss.str());
        }
        if (n_samples > 1){
            std::stringstream ss;
            ss << "Set bias with multiple samples is not supported!";
            throw std::runtime_error(ss.str());
        }
    

        dataHolder<const float> bias_holder{bias_ptr, stringTodeviceType(bias_device)};
        int output_dim = static_cast<int>(len(bias));
        py::gil_scoped_release release; 
        self.set_bias(&bias_holder, output_dim); 
    }, "Set GBRL model bias");
    gbrl.def("set_lambda_objs", [](GBRL &self, py::object &lambdas) {
        const float *lambdas_ptr = nullptr;
        std::vector<size_t> lambda_shape;
        std::string lambda_device;
        int n_samples, lambda_dim;

        handle_input_info<const float>(lambdas, lambdas_ptr, lambda_shape, lambda_device, "lambdas", false, "set_lambda_objs");

        if (lambda_shape.size() == 1){
            if (self.metadata->n_objs > 1){
                n_samples = 1;
                lambda_dim = static_cast<int>(lambda_shape[0]);
            } else{
                n_samples = static_cast<int>(lambda_shape[0]);
                lambda_dim = 1;
            }

            if (n_samples > 1){
                std::stringstream ss;
                ss << "Set lambdas with multiple samples is not supported!";
                throw std::runtime_error(ss.str());
            }
        } else {
            n_samples = static_cast<int>(lambda_shape[0]);
            lambda_dim = static_cast<int>(lambda_shape[1]);

            if (n_samples == self.metadata->n_objs && lambda_dim == 1){
                // Transpose case
                n_samples = 1;
                lambda_dim = static_cast<int>(lambda_shape[0]);
            }
        }
        if (lambda_dim != self.metadata->n_objs){
            std::stringstream ss;
            ss << "Targets dimension " << lambda_dim << " != correct number of objectives " << self.metadata->n_objs;
            throw std::runtime_error(ss.str());
        }
        if (n_samples > 1){
            std::stringstream ss;
            ss << "Set lambdas with multiple samples is not supported!";
            throw std::runtime_error(ss.str());
        }
    

        dataHolder<const float> lambda_holder{lambdas_ptr, stringTodeviceType(lambda_device)};
        int n_objs = static_cast<int>(len(lambdas));
        py::gil_scoped_release release; 
        self.set_lambda_objs(&lambda_holder, n_objs); 
    }, "Set GBRL model bias");
    
    // Set per-feature importance weights
    // Supports both NumPy arrays and PyTorch tensors on CPU or GPU
    // Automatically handles device-to-device transfers (CPU<->GPU)
    gbrl.def("set_feature_weights", [](GBRL &self, py::object &feature_weights) {
        const float *feature_weights_ptr = nullptr;
        std::vector<size_t> feature_weights_shape;
        std::string feature_weights_device;
        int n_samples, feature_weights_dim;

        handle_input_info<const float>(feature_weights, feature_weights_ptr, feature_weights_shape, feature_weights_device, "feature_weights", false, "set_feature_weights");

        if (feature_weights_shape.size() == 1){
            if (self.metadata->input_dim > 1){
                n_samples = 1;
                feature_weights_dim = static_cast<int>(feature_weights_shape[0]);
            } else{
                n_samples = static_cast<int>(feature_weights_shape[0]);
                feature_weights_dim = 1;
            }

            if (n_samples > 1){
                std::stringstream ss;
                ss << "Set feature_weights with multiple samples is not supported!";
                throw std::runtime_error(ss.str());
            }
        } else {
            n_samples = static_cast<int>(feature_weights_shape[0]);
            feature_weights_dim = static_cast<int>(feature_weights_shape[1]);

            if (n_samples == self.metadata->output_dim && feature_weights_dim == 1){
                // Transpose case
                n_samples = 1;
                feature_weights_dim = static_cast<int>(feature_weights_shape[0]);
            }
        }
        if (feature_weights_dim != self.metadata->input_dim){
            std::stringstream ss;
            ss << "Feature weights input dim " << feature_weights_dim << " != correct input dim " << self.metadata->input_dim;
            throw std::runtime_error(ss.str());
        }
        if (n_samples > 1){
            std::stringstream ss;
            ss << "Set feature_weights with multiple samples is not supported!";
            throw std::runtime_error(ss.str());
        }


        // Note: set_feature_weights only reads the data, so const_cast is safe here
        dataHolder<float> feature_weights_holder{const_cast<float*>(feature_weights_ptr), stringTodeviceType(feature_weights_device)};
        int input_dim = static_cast<int>(len(feature_weights));
        py::gil_scoped_release release;
        self.set_feature_weights(&feature_weights_holder, input_dim);
    }, "Set GBRL model feature weights");
    
    // Set monotonic constraints for model outputs
    // Configures per-feature monotonicity constraints: each feature can be constrained
    // to be monotonically increasing (+1) or decreasing (-1) for specific output dimensions.
    // Populates internal constraint arrays used during tree fitting and prediction.
    gbrl.def("set_monotonic_constraints", [](GBRL &self, const py::array_t<int> &feature_indices, const py::array_t<int> &output_indices, const py::array_t<int>& constraints) {
        if (!feature_indices.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("feature_indices must be C-contiguous");
        }
        if (!output_indices.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("output_indices must be C-contiguous");
        }
        if (!constraints.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("constraints must be C-contiguous");
        }

        // Get buffer info while holding GIL and validate 1D arrays
        py::buffer_info feature_info = feature_indices.request();
        if (feature_info.ndim != 1) {
            throw std::runtime_error("feature_indices must be a 1D array");
        }
        int* feature_indices_ptr = static_cast<int*>(feature_info.ptr);
        int n_constraints = static_cast<int>(feature_info.size);

        py::buffer_info output_info = output_indices.request();
        if (output_info.ndim != 1) {
            throw std::runtime_error("output_indices must be a 1D array");
        }
        int* output_indices_ptr = static_cast<int*>(output_info.ptr);
        if (static_cast<int>(output_info.size) != n_constraints) {
            throw std::runtime_error("feature_indices and output_indices must have the same length");
        }

        py::buffer_info constraints_info = constraints.request();
        if (constraints_info.ndim != 1) {
            throw std::runtime_error("constraints must be a 1D array");
        }
        int* constraints_ptr = static_cast<int*>(constraints_info.ptr);
        if (static_cast<int>(constraints_info.size) != n_constraints) {
            throw std::runtime_error("feature_indices and constraints must have the same length");
        }
        
        py::gil_scoped_release release; 
        self.set_monotonic_constraints(feature_indices_ptr, output_indices_ptr, constraints_ptr, n_constraints); 
    }, "Set GBRL model monotonic constraints");
    
    // Set feature mapping for mixed categorical/numerical inputs
    // Creates 4 arrays: feature_mapping, mapping_numerics (stored for export),
    // reverse_num_feature_mapping, reverse_cat_feature_mapping (used in computation)
    gbrl.def("set_feature_mapping", [](GBRL &self, const py::array_t<int> &feature_mapping, const py::array_t<bool> &mapping_numerics) {
        if (!feature_mapping.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        if (!mapping_numerics.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }

        // Get buffer info while holding GIL
        py::buffer_info info = feature_mapping.request();
        int* feature_mapping_ptr = static_cast<int*>(info.ptr);
        int input_dim = static_cast<int>(len(feature_mapping));

        info = mapping_numerics.request();
        bool* mapping_numerics_ptr = static_cast<bool*>(info.ptr);
        py::gil_scoped_release release; 
        self.set_feature_mapping(feature_mapping_ptr, mapping_numerics_ptr, input_dim); 
    }, "Set GBRL model feature mapping");
    gbrl.def("get_bias", [](GBRL &self) -> py::array_t<float> {
        py::gil_scoped_release release; 
        float* bias_ptr = self.get_bias();  
        int size = self.metadata->output_dim; // You need to know the size of the array
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(bias_ptr, [](void* ptr) {
            delete[] reinterpret_cast<float*>(ptr);});
        return py::array(size, bias_ptr, capsule);
    }, "Get GBRL model bias");
    gbrl.def("get_feature_weights", [](GBRL &self) -> py::array_t<float> {
        py::gil_scoped_release release; 
        float* feature_weights_ptr = self.get_feature_weights();  
        int size = self.metadata->input_dim; // You need to know the size of the array
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(feature_weights_ptr, [](void* ptr) {
            delete[] reinterpret_cast<float*>(ptr);});
        return py::array(size, feature_weights_ptr, capsule);
    }, "Get GBRL model feature weights");
    gbrl.def("get_feature_mapping", [](GBRL &self) -> py::tuple {
        py::gil_scoped_release release;
        int* feature_mapping_ptr = nullptr;
        bool* mapping_numerics_ptr = nullptr;
        self.get_feature_mapping(feature_mapping_ptr, mapping_numerics_ptr);
        int size = self.metadata->input_dim;
        py::gil_scoped_acquire acquire;
        
        auto feature_mapping_capsule = py::capsule(feature_mapping_ptr, [](void* ptr) {
            delete[] reinterpret_cast<int*>(ptr);
        });
        auto mapping_numerics_capsule = py::capsule(mapping_numerics_ptr, [](void* ptr) {
            delete[] reinterpret_cast<bool*>(ptr);
        });
        
        return py::make_tuple(
            py::array(size, feature_mapping_ptr, feature_mapping_capsule),
            py::array(size, mapping_numerics_ptr, mapping_numerics_capsule)
        );
    }, "Get GBRL model feature mapping (returns tuple of (feature_mapping, mapping_numerics))");
    gbrl.def("get_optimizers", [](GBRL &self) -> py::list {
        return getOptimizerConfigs(self.opts);
    }, "Get GBRL optimizers");
    gbrl.def("set_optimizer", [](GBRL &self, const std::string& algo, const std::string& scheduler_func, float init_lr, int start_idx, int stop_idx,
                                float stop_lr, int T, float beta_1, float beta_2, float eps, float shrinkage) {
        py::gil_scoped_release release; 
        self.set_optimizer(stringToAlgoType(algo), stringToSchedulerType(scheduler_func), init_lr, start_idx, stop_idx, stop_lr, T, beta_1, beta_2, eps, shrinkage); 
    }, py::arg("algo")="SGD", py::arg("scheduler")="const", py::arg("init_lr")=1.0, py::arg("start_idx")=0, py::arg("stop_idx")=0,
       py::arg("stop_lr")=1.0e-8, py::arg("T")=10000, py::arg("beta_1")=0.9, py::arg("beta_2")=0.999, 
       py::arg("eps")=1.0e-8, py::arg("shrinkage")=0.0,
       "Set optimizer!");
    // Predict method
    gbrl.def("predict", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::object start_tree_obj, py::object stop_tree_obj, bool return_torch) -> py::object {
        const float* obs_ptr = nullptr;
        const char* cat_obs_ptr = nullptr;
        std::vector<size_t> obs_shape, cat_obs_shape;
        std::string obs_device, cat_obs_device;
        int n_samples = 0, n_num_features = 0, n_cat_features = 0;
        
        // Handle start_tree_idx and stop_tree_idx - set to 0 if None
        int start_tree_idx = start_tree_obj.is_none() ? 0 : start_tree_obj.cast<int>();
        int stop_tree_idx = stop_tree_obj.is_none() ? 0 : stop_tree_obj.cast<int>();

        if (start_tree_idx < 0 || (start_tree_idx >= self.metadata->n_trees) && (self.metadata->n_trees > 0)) {
            std::stringstream ss;
            ss << "start_tree_idx is out of bounds! Got " << start_tree_idx 
               << ", but valid range is [0, " << self.metadata->n_trees - 1 << "]";
            throw std::runtime_error(ss.str());
        }
        if (stop_tree_idx < 0 || stop_tree_idx > self.metadata->n_trees) {
            std::stringstream ss;
            ss << "stop_tree_idx is out of bounds! Got " << stop_tree_idx 
               << ", but valid range is [0, " << self.metadata->n_trees << "]";
            throw std::runtime_error(ss.str());
        }

        handle_input_info<const float>(obs, obs_ptr, obs_shape, obs_device, "obs", true, "predict");
        handle_input_info<const char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "predict", CAT_TYPE);
        
        if (cat_obs_ptr == nullptr && obs_ptr == nullptr) {
            throw std::runtime_error("Cannot call predict without observations!");
        }

        if (obs_ptr != nullptr && cat_obs_ptr != nullptr) {
            if (obs_shape.size() == 1 && cat_obs_shape.size() == 1) {
                if (static_cast<int>(obs_shape[0]) + static_cast<int>(cat_obs_shape[0]) == self.metadata->input_dim) {
                    n_samples = 1;
                    n_num_features = static_cast<int>(obs_shape[0]);
                    n_cat_features = static_cast<int>(cat_obs_shape[0]);
                } else {
                    if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])) {
                        std::stringstream ss;
                        ss << "Number of samples is not equal between obs and categorical obs " 
                           << obs_shape[0] << " != " << cat_obs_shape[0];
                        throw std::runtime_error(ss.str());
                    }
                    n_samples = static_cast<int>(obs_shape[0]);
                    n_num_features = 1;
                    n_cat_features = 1;
                    if (n_num_features + n_cat_features != self.metadata->input_dim) {
                        std::stringstream ss;
                        ss << "Total number of features " << n_num_features + n_cat_features 
                           << " != input dim " << self.metadata->input_dim;
                        throw std::runtime_error(ss.str());
                    }
                }
            } else if (obs_shape.size() == 1) {
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])) {
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " 
                       << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = 1;
                n_cat_features = static_cast<int>(cat_obs_shape[1]);
            } else if (cat_obs_shape.size() == 1) {
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])) {
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " 
                       << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                n_cat_features = 1;
            } else {
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])) {
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " 
                       << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                n_cat_features = static_cast<int>(cat_obs_shape[1]);
            }
        } else if (obs_ptr != nullptr) {
            if (obs_shape.size() == 1) {
                if (static_cast<int>(obs_shape[0]) == self.metadata->input_dim) {
                    n_samples = 1;
                    n_num_features = static_cast<int>(obs_shape[0]);
                } else {
                    n_samples = static_cast<int>(obs_shape[0]);
                    n_num_features = 1;
                    if (n_num_features != self.metadata->input_dim) {
                        std::stringstream ss;
                        ss << "Total number of features " << n_num_features 
                           << " != input dim " << self.metadata->input_dim;
                        throw std::runtime_error(ss.str());
                    }
                }
            } else {
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                if (n_num_features != self.metadata->input_dim) {
                    std::stringstream ss;
                    ss << "Total number of features " << n_num_features 
                       << " != input dim " << self.metadata->input_dim;
                    throw std::runtime_error(ss.str());
                }
            }
        } else {
            if (cat_obs_shape.size() == 1) {
                if (static_cast<int>(cat_obs_shape[0]) == self.metadata->input_dim) {
                    n_samples = 1;
                    n_cat_features = static_cast<int>(cat_obs_shape[0]);
                } else {
                    n_samples = static_cast<int>(cat_obs_shape[0]);
                    n_cat_features = 1;
                    if (n_cat_features != self.metadata->input_dim) {
                        std::stringstream ss;
                        ss << "Total number of features " << n_cat_features 
                           << " != input dim " << self.metadata->input_dim;
                        throw std::runtime_error(ss.str());
                    }
                }
            } else {
                n_samples = static_cast<int>(cat_obs_shape[0]);
                n_cat_features = static_cast<int>(cat_obs_shape[1]);
                if (n_cat_features != self.metadata->input_dim){
                    std::stringstream ss;
                    ss << "Total number of features " << n_cat_features << " != input dim " << self.metadata->input_dim;
                    throw std::runtime_error(ss.str());
                }
            }
        }
        
        dataHolder<const float> obs_handler{obs_ptr, stringTodeviceType(obs_device)};
        dataHolder<const char> cat_obs_handler{cat_obs_ptr,stringTodeviceType(cat_obs_device)};

        if (n_cat_features + n_num_features != self.metadata->input_dim){
            std::stringstream ss;
            ss << "Total number of features " << n_cat_features + n_num_features << " != correct input dim " << self.metadata->input_dim;
            throw std::runtime_error(ss.str());
        }

        py::gil_scoped_release release; 
        float* result_ptr = self.predict(&obs_handler, &cat_obs_handler, n_samples, n_num_features, n_cat_features, start_tree_idx, stop_tree_idx);
        py::gil_scoped_acquire acquire;
        return return_tensor_info(n_samples, self.metadata->output_dim, result_ptr, self.device, return_torch);
    }, py::arg("obs"), py::arg("categorical_obs"), py::arg("start_tree_idx")=0, py::arg("stop_tree_idx")=0, py::arg("return_torch")=false, "Predict using the model");
        // saveToFile method
    gbrl.def("save", [](GBRL &self, const std::string& filename) -> int {
        py::gil_scoped_release release; 
        return self.saveToFile(filename); 
    }, "Save the model to a file");
    gbrl.def("export", [](GBRL &self, const std::string& filename, const std::string& modelname, const std::string& export_format, const std::string &export_type, const std::string& prefix) -> int {
        py::gil_scoped_release release; 
        return self.exportModel(filename, modelname, export_format, export_type, prefix); 
    }, py::arg("filename"), py::arg("modelname") = "", py::arg("export_format") = "float", py::arg("export_type") = "full", py::arg("prefix") = "", "Export model as a C-header file");
    gbrl.def("get_scheduler_lrs", [](GBRL &self) ->  py::array_t<float> {
        py::gil_scoped_release release; 
        float* lrs = self.get_scheduler_lrs(); 
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(lrs, [](void* ptr) {
        delete[] reinterpret_cast<float*>(ptr);
        });
        return py::array(static_cast<long int>(self.opts.size()), lrs, capsule);
    }, "Return current scheduler lrs");  
    gbrl.def("get_num_trees", [](GBRL &self) ->  int {
        py::gil_scoped_release release; 
        return self.get_num_trees(); 
    }, "Return current number of trees in the ensemble");  
    gbrl.def("get_metadata", [](GBRL &self) ->  py::dict {
        return metadataToDict(self.metadata); 
    }, "Return ensemble metadata");  
    gbrl.def("get_ensemble_data", [](GBRL &self) -> py::dict {
        py::gil_scoped_release release; 
        ensembleData *edata = self.get_ensemble_data(); 
        py::gil_scoped_acquire acquire;
        return ensembleDataToDict(edata, self.metadata);
    }, "Return ensemble data");
    gbrl.def("get_export_data", [](GBRL &self) -> py::dict {
        py::gil_scoped_release release; 
        exportData *exp_data = self.get_ensemble_export_data(); 
        py::gil_scoped_acquire acquire;
        py::dict result = ensembleExportDataToDict(exp_data);
        // Arrays are now owned by NumPy capsules; only delete the struct shell
        delete exp_data;
        return result;
    }, "Return export data with optimizer-scaled leaf values for inference");
    gbrl.def("get_tree", [](GBRL &self, int tree_idx) -> py::dict {
        py::gil_scoped_release release;
        treeData *tdata = self.get_tree(tree_idx);
        py::gil_scoped_acquire acquire;
        py::dict result = treeDataToDict(tdata);
        // Arrays are now owned by NumPy capsules; only delete the struct shell
        delete tdata;
        return result;
    }, py::arg("tree_idx"),
       "Extract a single tree from the ensemble as a dict of NumPy arrays.\n\n"
       "Parameters\n----------\n"
       "tree_idx : int\n    0-based index of the tree to extract.\n\n"
       "Returns\n-------\n"
       "dict\n    Dictionary with tree structure arrays (feature_indices, feature_values,\n"
       "    values, edge_weights, etc.) and scalar metadata.");
    gbrl.def("add_tree", [](GBRL &self, const py::dict& tree_dict) {
        treeData *tdata = dictToTreeData(tree_dict);
        {
            py::gil_scoped_release release;
            self.add_tree(tdata);
        }
        tree_data_dealloc(tdata);
    }, py::arg("tree_dict"),
       "Add a tree to the ensemble from a dict of NumPy arrays.\n\n"
       "Parameters\n----------\n"
       "tree_dict : dict\n    Dictionary as returned by get_tree(), containing tree structure\n"
       "    arrays and scalar metadata. Arrays are copied into the ensemble.\n\n"
       "Notes\n-----\n"
       "This is the counterpart to get_tree() for distributed (e.g. MPI)\n"
       "ensemble synchronization. Updates n_trees, n_leaves, and iteration.");
    gbrl.def("get_device", [](GBRL &self) ->  std::string {
        py::gil_scoped_release release; 
        return self.get_device(); 
    }, "Return the current device type");  
    gbrl.def("get_learner_name", [](GBRL &self) ->  std::string {
        py::gil_scoped_release release; 
        return self.get_learner_name(); 
    }, "Return the learner name");  
    gbrl.def("get_iteration", [](GBRL &self) ->  int {
        py::gil_scoped_release release; 
        return self.get_iteration(); 
    }, "Return current ensemble iteration");  
    gbrl.def("print_tree", [](GBRL &self, int tree_idx) {
        py::gil_scoped_release release; 
        self.print_tree(tree_idx); 
    }, py::arg("tree_idx") = -1, "Print specified tree index");
gbrl.def("get_matrix_representation", [](GBRL &self, py::object &obs, py::object &categorical_obs){
        const float* obs_ptr = nullptr;
        int n_num_features = 0;
        int n_samples = 0;
        int n_obs_samples = 0;
        
        if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Observation arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<const float*>(info_obs.ptr);
            
            if (info_obs.shape.size() == 1) {
                // 1D array - could be single sample with multiple features or multiple samples with 1 feature
                if (static_cast<int>(info_obs.shape[0]) == self.metadata->input_dim) {
                    n_samples = 1;
                    n_num_features = static_cast<int>(info_obs.shape[0]);
                } else {
                    n_samples = static_cast<int>(info_obs.shape[0]);
                    n_num_features = 1;
                }
                n_obs_samples = n_samples;
            } else {
                n_obs_samples = static_cast<int>(info_obs.shape[0]);
                n_num_features = static_cast<int>(info_obs.shape[1]);
                n_samples = n_obs_samples;
            }
        }
        
        int n_cat_features = 0;
        const char *cat_obs_ptr = nullptr;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Categorical observation arrays must be C-contiguous");
            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            
            if (info_categorical_obs.shape.size() == 1) {
                // 1D array - could be single sample or multiple samples with 1 feature
                int cat_size = static_cast<int>(info_categorical_obs.shape[0]);
                if (obs_ptr == nullptr) {
                    // Only categorical features
                    if (cat_size == self.metadata->input_dim) {
                        n_samples = 1;
                        n_cat_features = cat_size;
                    } else {
                        n_samples = cat_size;
                        n_cat_features = 1;
                    }
                } else {
                    // Have both numerical and categorical
                    if (cat_size == n_obs_samples) {
                        n_cat_features = 1;
                    } else if (n_obs_samples == 1) {
                        n_cat_features = cat_size;
                        n_samples = 1;
                    } else {
                        std::stringstream ss;
                        ss << "Categorical observation dimension mismatch: got " << cat_size 
                           << " but expected " << n_obs_samples << " samples";
                        throw std::runtime_error(ss.str());
                    }
                }
            } else {
                int n_cat_samples = static_cast<int>(info_categorical_obs.shape[0]);
                n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
                
                if (obs_ptr != nullptr && n_cat_samples != n_obs_samples) {
                    std::stringstream ss;
                    ss << "Number of categorical observation samples (" << n_cat_samples 
                       << ") != number of numerical observation samples (" << n_obs_samples << ")";
                    throw std::runtime_error(ss.str());
                }
                if (obs_ptr == nullptr) {
                    n_samples = n_cat_samples;
                }
            }
        }
        
        // Validate total feature count
        if (obs_ptr == nullptr && cat_obs_ptr == nullptr) {
            throw std::runtime_error("Cannot call get_matrix_representation without observations!");
        }
        
        if (n_cat_features + n_num_features != self.metadata->input_dim) {
            std::stringstream ss;
            ss << "Total number of features (" << n_cat_features + n_num_features 
               << ") != model input_dim (" << self.metadata->input_dim << ")";
            throw std::runtime_error(ss.str());
        }
        
        py::gil_scoped_release release; 
        matrixRepresentation *matrix = self.get_matrix_representation(obs_ptr, cat_obs_ptr, n_samples, n_num_features, n_cat_features);  
        py::gil_scoped_acquire acquire;
       
        auto capsule_A = py::capsule(matrix->A, [](void* ptr) {
            delete[] reinterpret_cast<bool*>(ptr);
        });
        auto capsule_V = py::capsule(matrix->V, [](void* ptr) {
            delete[] reinterpret_cast<float*>(ptr);
        });
        auto capsule_n_leaves_per_tree = py::capsule(matrix->n_leaves_per_tree, [](void* ptr) {
            delete[] reinterpret_cast<int*>(ptr);
        });
        auto np_array_A = py::array_t<bool>({n_samples, matrix->n_leaves + 1}, matrix->A, capsule_A);
        auto np_array_V = py::array_t<float>({matrix->n_leaves + 1, self.metadata->output_dim}, matrix->V, capsule_V);
        auto np_array_n_leaves_per_tree = py::array_t<int>({matrix->n_trees}, matrix->n_leaves_per_tree, capsule_n_leaves_per_tree);
        auto matrix_tuple = py::make_tuple(np_array_A, np_array_V, np_array_n_leaves_per_tree, matrix->n_leaves, matrix->n_trees);
        delete matrix;
        return matrix_tuple;
    }, py::arg("obs"), py::arg("categorical_obs"), "Get matrix representation of model given an input");
    gbrl.def("compress", [](GBRL &self, const int n_compressed_leaves, const int n_compressed_trees, py::object &leaf_indices, py::object &tree_indices, py::object &new_tree_indices, py::object &W){
        // Validate input parameters
        if (n_compressed_leaves <= 0) {
            throw std::runtime_error("n_compressed_leaves must be positive");
        }
        if (n_compressed_trees <= 0) {
            throw std::runtime_error("n_compressed_trees must be positive");
        }
        if (n_compressed_trees > self.metadata->n_trees) {
            std::stringstream ss;
            ss << "n_compressed_trees (" << n_compressed_trees 
               << ") cannot exceed current number of trees (" << self.metadata->n_trees << ")";
            throw std::runtime_error(ss.str());
        }
        
        const int* leaf_indices_ptr = nullptr;
        if (!leaf_indices.is_none()) {
            py::array_t<int> leaf_indices_array = py::cast<py::array_t<int>>(leaf_indices);
            if (!leaf_indices_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("leaf_indices array must be C-contiguous");
            py::buffer_info leaf_info = leaf_indices_array.request();
            leaf_indices_ptr = static_cast<const int*>(leaf_info.ptr);
            
            // Validate size
            if (leaf_info.size != n_compressed_leaves) {
                std::stringstream ss;
                ss << "leaf_indices size (" << leaf_info.size 
                   << ") does not match n_compressed_leaves (" << n_compressed_leaves << ")";
                throw std::runtime_error(ss.str());
            }
        } else {
            throw std::runtime_error("leaf_indices cannot be None");
        }
        
        const int* tree_indices_ptr = nullptr;
        if (!tree_indices.is_none()) {
            py::array_t<int> tree_indices_array = py::cast<py::array_t<int>>(tree_indices);
            if (!tree_indices_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("tree_indices array must be C-contiguous");
            py::buffer_info tree_info = tree_indices_array.request();
            tree_indices_ptr = static_cast<const int*>(tree_info.ptr);
            
            // Validate size
            if (tree_info.size != n_compressed_trees) {
                std::stringstream ss;
                ss << "tree_indices size (" << tree_info.size 
                   << ") does not match n_compressed_trees (" << n_compressed_trees << ")";
                throw std::runtime_error(ss.str());
            }
        } else {
            throw std::runtime_error("tree_indices cannot be None");
        }
        
        const int* new_tree_indices_ptr = nullptr;
        if (!new_tree_indices.is_none()) {
            py::array_t<int> new_tree_indices_array = py::cast<py::array_t<int>>(new_tree_indices);
            if (!new_tree_indices_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("new_tree_indices array must be C-contiguous");
            py::buffer_info indices_info = new_tree_indices_array.request();
            new_tree_indices_ptr = static_cast<const int*>(indices_info.ptr);
            
            // Validate size
            if (indices_info.size != n_compressed_trees) {
                std::stringstream ss;
                ss << "new_tree_indices size (" << indices_info.size 
                   << ") does not match n_compressed_trees (" << n_compressed_trees << ")";
                throw std::runtime_error(ss.str());
            }
        } else {
            throw std::runtime_error("new_tree_indices cannot be None");
        }
        
        const float* W_ptr = nullptr;
        if (!W.is_none()) {
            py::array_t<float> W_array = py::cast<py::array_t<float>>(W);
            if (!W_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("W array must be C-contiguous");
            py::buffer_info w_info = W_array.request();
            W_ptr = static_cast<const float*>(w_info.ptr);
            
            // Validate shape (should be n_compressed_leaves+1 x output_dim)
            if (w_info.ndim != 2) {
                throw std::runtime_error("W must be a 2D array");
            }
            if (static_cast<int>(w_info.shape[0]) != n_compressed_leaves + 1 || 
                static_cast<int>(w_info.shape[1]) != self.metadata->output_dim) {
                std::stringstream ss;
                ss << "W shape (" << w_info.shape[0] << ", " << w_info.shape[1] 
                   << ") does not match expected (" << n_compressed_leaves + 1 
                   << ", " << self.metadata->output_dim << ")";
                throw std::runtime_error(ss.str());
            }
        } else {
            throw std::runtime_error("W correction matrix cannot be None");
        }
        
        py::gil_scoped_release release; 
        self.compress_ensemble(n_compressed_leaves, n_compressed_trees, leaf_indices_ptr, tree_indices_ptr, new_tree_indices_ptr, W_ptr);  

    }, py::arg("n_compressed_leaves"), py::arg("n_compressed_trees"), py::arg("leaf_indices"), py::arg("tree_indices"), py::arg("new_tree_indices"), py::arg("W") , "Compress ensemble");
    gbrl.def("tree_shap", [](GBRL &self, const int tree_idx, py::object &obs, py::object &categorical_obs, 
                            py::object &norm_values, py::object &base_poly, py::object &offset) -> py::array_t<float> {
        const float* obs_ptr = nullptr;
        int n_num_features = 0;
        int n_samples = 0;
        if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<const float*>(info_obs.ptr);
            if (info_obs.shape.size() == 1) {
                n_num_features = static_cast<int>(info_obs.shape[0]);
                n_samples = 1;
            } else {
                n_num_features = static_cast<int>(info_obs.shape[1]);
                n_samples = static_cast<int>(info_obs.shape[0]);
            }
        }

        int n_cat_features = 0;
        const char *cat_obs_ptr = nullptr;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");

            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            if (info_categorical_obs.shape.size() == 1) {
                n_cat_features = static_cast<int>(info_categorical_obs.shape[0]);
                if (n_samples == 0) n_samples = 1;
            } else {
                n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
                if (n_samples == 0) n_samples = static_cast<int>(info_categorical_obs.shape[0]);
            }
        }
        float *norm_ptr = nullptr;
        if (!norm_values.is_none()) {
            py::array_t<float> norm_array = py::cast<py::array_t<float>>(norm_values);
            if (!norm_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_norm = norm_array.request();
            norm_ptr = static_cast<float*>(info_norm.ptr);
        }
        float *base_poly_ptr = nullptr;
        if (!base_poly.is_none()) {
            py::array_t<float> base_poly_array = py::cast<py::array_t<float>>(base_poly);
            if (!base_poly_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_base_poly = base_poly_array.request();
            base_poly_ptr = static_cast<float*>(info_base_poly.ptr);
        }
        float *offset_ptr = nullptr;
        if (!offset.is_none()) {
            py::array_t<float> offset_array = py::cast<py::array_t<float>>(offset);
            if (!offset_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_offset = offset_array.request();
            offset_ptr = static_cast<float*>(info_offset.ptr);
        }
        py::gil_scoped_release release; 
        float* shap_values = self.tree_shap(tree_idx, obs_ptr, cat_obs_ptr, n_samples, norm_ptr, base_poly_ptr, offset_ptr);
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(shap_values, [](void* ptr) {
        delete[] reinterpret_cast<float*>(ptr);
        });
        return py::array({n_samples, n_num_features + n_cat_features, self.metadata->output_dim}, shap_values, capsule);
    }, py::arg("tree_idx")=0, py::arg("obs"), py::arg("categorical_obs"), py::arg("norm_values"), py::arg("base_poly"), py::arg("offset"), "Calculate SHAP values of a single tree");
    gbrl.def("ensemble_shap", [](GBRL &self, py::object &obs, py::object &categorical_obs, 
                            py::object &norm_values, py::object &base_poly, py::object &offset) -> py::array_t<float> {
        const float* obs_ptr = nullptr;
        int n_num_features = 0;
        int n_samples = 0;
        if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<const float*>(info_obs.ptr);
            if (info_obs.shape.size() == 1) {
                n_num_features = static_cast<int>(info_obs.shape[0]);
                n_samples = 1;
            } else {
                n_num_features = static_cast<int>(info_obs.shape[1]);
                n_samples = static_cast<int>(info_obs.shape[0]);
            }
        }

        int n_cat_features = 0;
        const char *cat_obs_ptr = nullptr;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");

            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            if (info_categorical_obs.shape.size() == 1) {
                n_cat_features = static_cast<int>(info_categorical_obs.shape[0]);
                if (n_samples == 0) n_samples = 1;
            } else {
                n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
                if (n_samples == 0) n_samples = static_cast<int>(info_categorical_obs.shape[0]);
            }
        }
        float *norm_ptr = nullptr;
        if (!norm_values.is_none()) {
            py::array_t<float> norm_array = py::cast<py::array_t<float>>(norm_values);
            if (!norm_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_norm = norm_array.request();
            norm_ptr = static_cast<float*>(info_norm.ptr);
        }
        float *base_poly_ptr = nullptr;
        if (!base_poly.is_none()) {
            py::array_t<float> base_poly_array = py::cast<py::array_t<float>>(base_poly);
            if (!base_poly_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_base_poly = base_poly_array.request();
            base_poly_ptr = static_cast<float*>(info_base_poly.ptr);
        }
        float *offset_ptr = nullptr;
        if (!offset.is_none()) {
            py::array_t<float> offset_array = py::cast<py::array_t<float>>(offset);
            if (!offset_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_offset = offset_array.request();
            offset_ptr = static_cast<float*>(info_offset.ptr);
        }
        py::gil_scoped_release release; 
        float* shap_values = self.ensemble_shap(obs_ptr, cat_obs_ptr, n_samples, norm_ptr, base_poly_ptr, offset_ptr);
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(shap_values, [](void* ptr) {
        delete[] reinterpret_cast<float*>(ptr);
        });
        return py::array({n_samples, n_num_features + n_cat_features, self.metadata->output_dim}, shap_values, capsule);
    }, py::arg("obs"), py::arg("categorical_obs"), py::arg("norm_values"), py::arg("base_poly"), py::arg("offset"), "Calculate SHAP values of a single tree");
    gbrl.def_static("cuda_available", &GBRL::cuda_available, "Return if CUDA is available"); 
    gbrl.def("plot_tree", [](GBRL &self, int tree_idx, const std::string &filename) {
        py::gil_scoped_release release; 
        self.plot_tree(tree_idx, filename); 
    }, py::arg("tree_idx") = -1,
       py::arg("filename"),
     "Plot specified tree index to png file"); 
    gbrl.def("print_ensemble_metadata", [](GBRL &self) {
        py::gil_scoped_release release; 
        self.print_ensemble_metadata(); 
    }, "Print ensemble metadata"); 
}