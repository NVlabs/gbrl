//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#ifdef USE_CUDA
#include <cuda_runtime.h>  // For cudaMalloc, cudaFree
#endif

#include "gbrl.h"
#include "types.h"

#include "dlpack/dlpack.h"

namespace py = pybind11;

template <typename T>
void get_numpy_array_info(py::object obj, T*& ptr, std::vector<size_t>& shape, const std::string& expected_format = ""){
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

template <typename T>
void get_tensor_info(py::tuple tensor_info, T*& ptr, std::vector<size_t>& shape, std::string& device) {
    if (tensor_info.size() != 4) {
        throw std::runtime_error("Expected a tuple of size 4: (data_ptr, shape, dtype, device)");
    }
    // size_t raw_ptr = tensor_info[0].cast<size_t>();
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
        throw std::runtime_error("Unsupported data type: " + dtype) ;
    }
    if (dtype != expected_dtype) {
        throw std::runtime_error("Expected dtype " + expected_dtype + ", but got " + dtype);
    }
    // Extract device
    device = tensor_info[3].cast<std::string>();
}

template <typename T>
void handle_input_info(py::object& input, T*& ptr, std::vector<size_t>& shape, std::string& device, const std::string& name, const bool none_allowed, const std::string& function_name, const std::string& expected_format = ""){
    if (input.is_none()) {
        if (!none_allowed)
            throw std::runtime_error("Cannot call " + function_name + " without " + name + "!");
        else{
            device = "cpu";
            ptr = nullptr;
            return;
        }
    } 
    if (py::isinstance<py::array>(input)) {
        get_numpy_array_info<T>(input, ptr, shape, expected_format);  // Handle NumPy array input
        device = "cpu";
    } else if (py::isinstance<py::tuple>(input)) {
        get_tensor_info<T>(input, ptr, shape, device);  // Handle tuple input
    } else {
        throw std::runtime_error("Unknown " + name + " type! Must be a NumPy array or tuple.");
    }
}

void dlpack_deleter_function(DLManagedTensor* self) {
#ifdef USE_CUDA
    if (self->dl_tensor.device.device_type == kDLCUDA) {
        cudaFree(static_cast<float*>(self->dl_tensor.data));  // GPU memory
    }
#endif 
    if (self->dl_tensor.device.device_type == kDLCPU) {
        delete[] static_cast<float*>(self->dl_tensor.data);  // CPU memory
    }
    delete[] self->dl_tensor.shape;
    delete self;
}


py::capsule create_dlpack_tensor(void* raw_ptr, const std::vector<int64_t>& shape, DLDataType dtype, DLDevice device) {
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
    // Create a PyCapsule and pass the DLTensor to Python
    managed_tensor->manager_ctx = nullptr;
    // Set the deleter function
    managed_tensor->deleter = dlpack_deleter_function;
    // Return the PyCapsule containing the DLManagedTensor
    return py::capsule(managed_tensor, "dltensor");
}

py::object return_tensor_info(int num_samples, int output_dim, float *ptr, deviceType device, bool is_torch) {
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
        d["iteration"] = metadata->iteration;
    }
    return d;
}

py::dict ensembleDataToDict(const ensembleData* data, const ensembleMetaData* metadata) {
    py::dict d;
    if (data != nullptr) {
        // Convert float pointers to NumPy arrays with ownership transfer
        auto bias_capsule = py::capsule(data->bias, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["bias"] = py::array_t<float>({metadata->output_dim}, data->bias, bias_capsule);

        auto feature_weights_capsule = py::capsule(data->feature_weights, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["feature_weights"] = py::array_t<float>({metadata->input_dim}, data->feature_weights, feature_weights_capsule);

        auto tree_indices_capsule = py::capsule(data->tree_indices, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["tree_indices"] = py::array_t<int>({metadata->n_trees}, data->tree_indices, tree_indices_capsule);

        int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;

        auto depths_capsule = py::capsule(data->depths, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["depths"] = py::array_t<int>({split_sizes}, data->depths, depths_capsule);

        auto values_capsule = py::capsule(data->values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["values"] = py::array_t<float>({metadata->n_leaves, metadata->output_dim}, data->values, values_capsule);

        auto feature_indices_capsule = py::capsule(data->feature_indices, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["feature_indices"] = py::array_t<int>({split_sizes, metadata->max_depth}, data->feature_indices, feature_indices_capsule);

        auto feature_values_capsule = py::capsule(data->feature_values, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["feature_values"] = py::array_t<float>({split_sizes, metadata->max_depth}, data->feature_values, feature_values_capsule);

        auto edge_weights_capsule = py::capsule(data->edge_weights, [](void* ptr) { delete[] reinterpret_cast<float*>(ptr); });
        d["edge_weights"] = py::array_t<float>({metadata->n_leaves, metadata->max_depth}, data->edge_weights, edge_weights_capsule);

        auto is_numerics_capsule = py::capsule(data->is_numerics, [](void* ptr) { delete[] reinterpret_cast<bool*>(ptr); });
        d["is_numerics"] = py::array_t<bool>({split_sizes, metadata->max_depth}, data->is_numerics, is_numerics_capsule);

        auto inequality_directions_capsule = py::capsule(data->inequality_directions, [](void* ptr) { delete[] reinterpret_cast<bool*>(ptr); });
        d["inequality_directions"] = py::array_t<bool>({metadata->n_leaves, metadata->max_depth}, data->inequality_directions, inequality_directions_capsule);
      
        // Convert char* categorical_values to NumPy string array (S128)
        auto categorical_capsule = py::capsule(data->categorical_values, [](void* ptr) { delete[] reinterpret_cast<char*>(ptr); });
        d["categorical_values"] = py::array(py::dtype("S128"), {split_sizes, metadata->max_depth}, data->categorical_values, categorical_capsule);
        
        d["alloc_data_size"] = data->alloc_data_size;

#ifdef DEBUG
        auto n_samples_capsule = py::capsule(data->n_samples, [](void* ptr) { delete[] reinterpret_cast<int*>(ptr); });
        d["n_samples"] = py::array_t<int>({metadata->n_leaves}, data->n_samples, n_samples_capsule);
#endif
    }
    return d;
}


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

py::list getOptimizerConfigs(const std::vector<Optimizer*>& opts) {
    py::list configs;
    for (auto& opt : opts) {
        optimizerConfig* conf = opt->getConfig();
        configs.append(optimizerToDict(conf));  // conf is deleted within optimizerConfigToDict
    }
    return configs;
}

PYBIND11_MODULE(gbrl_cpp, m) {
    py::class_<GBRL> gbrl(m, "GBRL");
    gbrl.def(py::init<int, int, int, int, int, int, int, float, std::string, std::string, bool, int, std::string, int, std::string>(),
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
         py::arg("verbose")=0,
         py::arg("device")="cpu",
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
    gbrl.def("step", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::object &grads) {
        const float* obs_ptr = nullptr;
        const char* cat_obs_ptr= nullptr;
        float* grads_ptr = nullptr;
        std::vector<size_t> obs_shape, cat_obs_shape, grads_shape;
        std::string obs_device, cat_obs_device, grads_device;
        int n_samples, n_num_features = 0, n_cat_features = 0, grad_output_dim;
        int n_obs_samples, n_cat_samples;
        
        handle_input_info<float>(grads, grads_ptr, grads_shape, grads_device, "grads", false, "step");
        if (grads_shape.size() == 1){
            if (self.metadata->output_dim > 1){
                n_samples = 1;
                grad_output_dim = static_cast<int>(grads_shape[0]);
            } else{
                n_samples = static_cast<int>(grads_shape[0]);
                grad_output_dim = 1;
            }
        } else {
            n_samples = static_cast<int>(grads_shape[0]);
            grad_output_dim = static_cast<int>(grads_shape[1]);
        }
        if (grad_output_dim != self.metadata->output_dim){
                std::stringstream ss;
                ss << "Gradient output dim " << grad_output_dim << " != correct output dim " << self.metadata->output_dim;
                throw std::runtime_error(ss.str());
            }

        dataHolder<float> grads_handler{grads_ptr, stringTodeviceType(grads_device)};

        handle_input_info<const float>(obs, obs_ptr, obs_shape, obs_device, "obs", true, "step");
        if (obs_ptr != nullptr){
            if (n_samples == 1){
               n_num_features = static_cast<int>(obs_shape[0]);
               if (obs_shape.size() > 1){
                    std::stringstream ss;
                    ss << "gradients has 1 sample but observations have multiple.";
                    throw std::runtime_error(ss.str());
               }
            } else {
                n_obs_samples = static_cast<int>(obs_shape[0]);
                n_num_features = (obs_shape.size() == 1) ? 1 : static_cast<int>(obs_shape[1]);
                if (n_obs_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of observations " << n_obs_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            }
        }

        dataHolder<const float> obs_handler{obs_ptr, stringTodeviceType(obs_device)};

        handle_input_info<const char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "step", CAT_TYPE);
        if (cat_obs_ptr != nullptr){
            if (n_samples == 1){
               n_cat_features = static_cast<int>(cat_obs_shape[0]);
               if (cat_obs_shape.size() > 1){
                    std::stringstream ss;
                    ss << "gradients has 1 sample but categorical observations have multiple.";
                    throw std::runtime_error(ss.str());
               }
            } else {
                n_cat_samples = static_cast<int>(cat_obs_shape[0]);
                n_cat_features = (cat_obs_shape.size() == 1) ? 1 : static_cast<int>(cat_obs_shape[1]);
                if (n_cat_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of categorical observations " << n_cat_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            }
        }

        dataHolder<const char> cat_obs_handler{cat_obs_ptr, stringTodeviceType(cat_obs_device)};

        py::gil_scoped_release release;
        self.step(&obs_handler, &cat_obs_handler, &grads_handler, n_samples, n_num_features, n_cat_features);
    },  py::arg("obs"),
        py::arg("categorical_obs"),
        py::arg("grads"),
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
            if (n_samples == 1){
               n_num_features = static_cast<int>(obs_shape[0]);
               if (obs_shape.size() > 1){
                    std::stringstream ss;
                    ss << "gradients has 1 sample but observations have multiple.";
                    throw std::runtime_error(ss.str());
               }
            } else {
                n_obs_samples = static_cast<int>(obs_shape[0]);
                n_num_features = (obs_shape.size() == 1) ? 1 : static_cast<int>(obs_shape[1]);
                if (n_obs_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of observations " << n_obs_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            }
        }

        dataHolder<float> obs_handler{obs_ptr, stringTodeviceType(obs_device)};

        handle_input_info<char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "fit", CAT_TYPE);

        if (cat_obs_ptr != nullptr){
            if (n_samples == 1){
               n_cat_features = static_cast<int>(cat_obs_shape[0]);
               if (cat_obs_shape.size() > 1){
                    std::stringstream ss;
                    ss << "gradients has 1 sample but categorical observations have multiple.";
                    throw std::runtime_error(ss.str());
               }
            } else {
                n_cat_samples = static_cast<int>(cat_obs_shape[0]);
                n_cat_features = (cat_obs_shape.size() == 1) ? 1 : static_cast<int>(cat_obs_shape[1]);
                if (n_cat_samples != n_samples){
                    std::stringstream ss;
                    ss << "Number of categorical observations " << n_cat_samples << " != number of gradient samples " << n_samples;
                    throw std::runtime_error(ss.str());
                }
            }
        }

        dataHolder<char> cat_obs_handler{cat_obs_ptr, stringTodeviceType(cat_obs_device)};

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

        handle_input_info<const float>(bias, bias_ptr, bias_shape, bias_device, "bias", false, "set_bias");

        int n_samples = (bias_shape.size() == 1) ? 1 : static_cast<int>(bias_shape[0]);
        int bias_dim = (bias_shape.size() == 1) ? static_cast<int>(bias_shape[0]) : static_cast<int>(bias_shape[1]);
        if (n_samples > 1){
            std::stringstream ss;
            ss << "Bias should be a vector not a tensor -> 1 single sample";
            throw std::runtime_error(ss.str());
        }
        if (bias_dim != self.metadata->output_dim){
            std::stringstream ss;
            ss << "Bias dim " << bias_dim << " != correct output dim " << self.metadata->output_dim;
            throw std::runtime_error(ss.str());
        }

        dataHolder<const float> bias_holder{bias_ptr, stringTodeviceType(bias_device)};
        int output_dim = static_cast<int>(len(bias));
        py::gil_scoped_release release; 
        self.set_bias(&bias_holder, output_dim); 
    }, "Set GBRL model bias");
    gbrl.def("set_feature_weights", [](GBRL &self, const py::array_t<float> &feature_weights) {
        if (!feature_weights.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        
        py::buffer_info info = feature_weights.request(false);
        float* feature_weights_ptr = static_cast<float*>(info.ptr);
        int input_dim = static_cast<int>(len(feature_weights));
        py::gil_scoped_release release; 
        self.set_feature_weights(feature_weights_ptr, input_dim); 
    }, "Set GBRL model feature weights");
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
    }, "Get GBRL model bias");
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
   // predict method
    gbrl.def("predict", [](GBRL &self, py::object &obs, py::object &categorical_obs, int start_tree_idx, int stop_tree_idx, bool return_torch) -> py::object {
        const float* obs_ptr = nullptr;
        const char* cat_obs_ptr= nullptr;
        std::vector<size_t> obs_shape, cat_obs_shape;
        std::string obs_device, cat_obs_device;
        int n_samples = 0, n_num_features = 0, n_cat_features = 0;

        handle_input_info<const float>(obs, obs_ptr, obs_shape, obs_device, "obs", true, "predict");
        handle_input_info<const char>(categorical_obs, cat_obs_ptr, cat_obs_shape, cat_obs_device, "cat_obs", true, "predict", CAT_TYPE); 
        if (cat_obs_ptr == nullptr && obs_ptr == nullptr){
            throw std::runtime_error("Cannot call predict without observations!");
        }

        if (obs_ptr != nullptr && cat_obs_ptr != nullptr){
            if (obs_shape.size() == 1 && cat_obs_shape.size() == 1){
                if (static_cast<int>(obs_shape[0]) + static_cast<int>(cat_obs_shape[0]) == self.metadata->input_dim){
                    n_samples = 1;
                    n_num_features = static_cast<int>(obs_shape[0]);
                    n_cat_features = static_cast<int>(cat_obs_shape[0]);
                } else {
                    if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])){
                        std::stringstream ss;
                        ss << "Number of samples is not equal between obs and categorical obs " << obs_shape[0] << " != " << cat_obs_shape[0];
                        throw std::runtime_error(ss.str());
                    }
                    n_samples = static_cast<int>(obs_shape[0]);
                    n_num_features = 1;
                    n_cat_features = 1;
                    if (n_num_features + n_cat_features != self.metadata->input_dim){
                        std::stringstream ss;
                        ss << "Total number of features " << n_num_features + n_cat_features << " != input dim " << self.metadata->input_dim;
                        throw std::runtime_error(ss.str());
                    }
                }
            } else if (obs_shape.size() == 1){
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])){
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = 1;
                n_cat_features = static_cast<int>(cat_obs_shape[1]);
            } else if (cat_obs_shape.size() == 1){
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])){
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                n_cat_features = 1;
            } else {
                if (static_cast<int>(obs_shape[0]) != static_cast<int>(cat_obs_shape[0])){
                    std::stringstream ss;
                    ss << "Number of samples is not equal between obs and categorical obs " << obs_shape[0] << " != " << cat_obs_shape[0];
                    throw std::runtime_error(ss.str());
                }
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                n_cat_features = static_cast<int>(cat_obs_shape[1]);
            }
        } else if (obs_ptr != nullptr){
            if (obs_shape.size() == 1){
                if (static_cast<int>(obs_shape[0]) == self.metadata->input_dim){
                    n_samples = 1;
                    n_num_features = static_cast<int>(obs_shape[0]);
                } else {
                    n_samples = static_cast<int>(obs_shape[0]);
                    n_num_features = 1;
                    if (n_num_features != self.metadata->input_dim){
                        std::stringstream ss;
                        ss << "Total number of features " << n_num_features << " != input dim " << self.metadata->input_dim;
                        throw std::runtime_error(ss.str());
                    }
                }

            } else {
                n_samples = static_cast<int>(obs_shape[0]);
                n_num_features = static_cast<int>(obs_shape[1]);
                if (n_num_features != self.metadata->input_dim){
                    std::stringstream ss;
                    ss << "Total number of features " << n_num_features << " != input dim " << self.metadata->input_dim;
                    throw std::runtime_error(ss.str());
                }
            }
        } else {
            if (cat_obs_shape.size() == 1){
                if (static_cast<int>(cat_obs_shape[0]) == self.metadata->input_dim){
                    n_samples = 1;
                    n_cat_features = static_cast<int>(cat_obs_shape[0]);
                } else {
                    n_samples = static_cast<int>(cat_obs_shape[0]);
                    n_cat_features = 1;
                    if (n_cat_features != self.metadata->input_dim){
                        std::stringstream ss;
                        ss << "Total number of features " << n_cat_features << " != input dim " << self.metadata->input_dim;
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
    gbrl.def("get_ensemble_data", [](GBRL &self) -> py::dict{
        py::gil_scoped_release release; 
        ensembleData *edata = self.get_ensemble_data(); 
        py::gil_scoped_acquire acquire;
        return ensembleDataToDict(edata, self.metadata);
    }, "Return ensemble data");
    gbrl.def("get_device", [](GBRL &self) ->  std::string {
        py::gil_scoped_release release; 
        return self.get_device(); 
    }, "Return the current device type");  
    gbrl.def("get_iteration", [](GBRL &self) ->  int {
        py::gil_scoped_release release; 
        return self.get_iteration(); 
    }, "Return current ensemble iteration");  
    gbrl.def("print_tree", [](GBRL &self, int tree_idx) {
        py::gil_scoped_release release; 
        self.print_tree(tree_idx); 
    }, py::arg("tree_idx") = -1, "Print specified tree index");
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
            n_num_features = static_cast<int>(info_obs.shape[1]);
            n_samples = static_cast<int>(info_obs.shape[0]);
        }

        int n_cat_features = 0;
        const char *cat_obs_ptr = nullptr;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");

            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
            n_samples = static_cast<int>(info_categorical_obs.shape[0]);
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
            n_num_features = static_cast<int>(info_obs.shape[1]);
            n_samples = static_cast<int>(info_obs.shape[0]);
        }

        int n_cat_features = 0;
        const char *cat_obs_ptr = nullptr;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");

            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
            n_samples = static_cast<int>(info_categorical_obs.shape[0]);
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