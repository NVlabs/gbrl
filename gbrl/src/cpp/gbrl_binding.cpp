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

#include "gbrl.h"
#include "types.h"

namespace py = pybind11;

py::dict metadataToDict(const ensembleMetaData* metadata){
    py::dict d;
    if (metadata != nullptr){
        d["output_dim"] = metadata->output_dim;
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
    gbrl.def(py::init<int, int, int, int, int, float, std::string, std::string, bool, int, std::string, int, std::string>(),
         py::arg("output_dim")=1, 
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
    gbrl.def("step", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::array_t<float> &grads, py::array_t<float> &feature_weights) {
        if (!grads.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        py::buffer_info info_grads = grads.request();
        float* grads_ptr = static_cast<float*>(info_grads.ptr);
        int n_samples = static_cast<int>(info_grads.shape[0]);

        const float* obs_ptr = nullptr;
        int n_num_features = 0;
        if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<const float*>(info_obs.ptr);
            n_num_features = static_cast<int>(info_obs.shape[1]);
        }
        const char* cat_obs_ptr = nullptr;
        int n_cat_features = 0;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<const char*>(info_categorical_obs.ptr);
            n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
        }

        if (!feature_weights.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        py::buffer_info info_feature_weights = feature_weights.request();
        float* feature_weights_ptr = static_cast<float*>(info_feature_weights.ptr);
                    
        py::gil_scoped_release release; 
        self.step(obs_ptr, cat_obs_ptr, grads_ptr, feature_weights_ptr, n_samples, n_num_features, n_cat_features); 
    },  py::arg("obs"),
        py::arg("categorical_obs"),
        py::arg("grads"),
        py::arg("feature_weights"),
    "Fit a decision tree with the given observations and gradients");
    gbrl.def("fit", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::array_t<float> &targets, py::array_t<float> &feature_weights, int iterations, bool shuffle, std::string loss_type) -> float {
        if (!targets.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }

        float* obs_ptr = nullptr;
        int n_num_features = 0;
        if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<float*>(info_obs.ptr);
            n_num_features = static_cast<int>(info_obs.shape[1]);
        }

        char* cat_obs_ptr = nullptr;
        int n_cat_features = 0;
        if (!categorical_obs.is_none()) {
            py::array py_array = py::cast<py::array>(categorical_obs);
            if (!py_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_categorical_obs = py_array.request();
            cat_obs_ptr = static_cast<char*>(info_categorical_obs.ptr);
            n_cat_features = static_cast<int>(info_categorical_obs.shape[1]);
        }
        
        if (!feature_weights.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        py::buffer_info info_feature_weights = feature_weights.request();
        float* feature_weights_ptr = static_cast<float*>(info_feature_weights.ptr);
        
        py::gil_scoped_release release; 
        py::buffer_info info_targets = targets.request();
        float* targets_ptr = static_cast<float*>(info_targets.ptr);
        int n_samples = static_cast<int>(info_targets.shape[0]);
        return self.fit(obs_ptr, cat_obs_ptr, targets_ptr, feature_weights_ptr, iterations, n_samples, n_num_features, n_cat_features, shuffle, loss_type); 
    },  py::arg("obs"),
        py::arg("categorical_obs"),
        py::arg("targets"),
        py::arg("feature_weights"),
        py::arg("iterations"),
        py::arg("shuffle")=true,  
        py::arg("loss_type")="MultiRMSE",  
    "Fit a decision tree with the given observations and targets for <iterations> boosting rounds");
    gbrl.def("set_bias", [](GBRL &self, const py::array_t<float> &bias) {
        if (!bias.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        py::gil_scoped_release release; 

        py::buffer_info info = bias.request();
        float* bias_ptr = static_cast<float*>(info.ptr);
        int output_dim = static_cast<int>(len(bias));
        
        self.set_bias(bias_ptr, output_dim); 
    }, "Set GBRL model bias");
    gbrl.def("get_bias", [](GBRL &self) -> py::array_t<float> {
        py::gil_scoped_release release; 
        float* bias_ptr = self.get_bias();  
        int size = self.metadata->output_dim; // You need to know the size of the array
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(bias_ptr, [](void* ptr) {
            delete[] reinterpret_cast<float*>(ptr);});
        return py::array(size, bias_ptr, capsule);
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
    gbrl.def("predict", [](GBRL &self, py::object &obs, py::object &categorical_obs, int start_tree_idx, int stop_tree_idx) -> py::array_t<float> {
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

        py::gil_scoped_release release; 
        float* result_ptr = self.predict(obs_ptr, cat_obs_ptr, n_samples, n_num_features, n_cat_features, start_tree_idx, stop_tree_idx);
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(result_ptr, [](void* ptr) {
        delete[] reinterpret_cast<float*>(ptr);
        });
        return py::array({n_samples, self.metadata->output_dim}, result_ptr, capsule);
    }, py::arg("obs"), py::arg("categorical_obs"), py::arg("start_tree_idx")=0, py::arg("stop_tree_idx")=0, "Predict using the model");
    gbrl.def("predict", [](GBRL &self, py::object &obs, py::object &categorical_obs, py::array_t<float> start_preds, int start_tree_idx, int stop_tree_idx){
        if (!start_preds.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw std::runtime_error("Arrays must be C-contiguous");
        }
        const float* obs_ptr = nullptr;
        int n_num_features = 0;
         if (!obs.is_none()) {
            py::array_t<float> obs_array = py::cast<py::array_t<float>>(obs);
            if (!obs_array.attr("flags").attr("c_contiguous").cast<bool>())
                throw std::runtime_error("Arrays must be C-contiguous");
            py::buffer_info info_obs = obs_array.request();
            obs_ptr = static_cast<const float*>(info_obs.ptr);
            n_num_features = static_cast<int>(info_obs.shape[1]);
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
        }
        py::gil_scoped_release release; 
        py::buffer_info info_preds = start_preds.request();
        float* preds_ptr = static_cast<float*>(info_preds.ptr);
        int n_samples = static_cast<int>(info_preds.shape[0]);
        self.predict(obs_ptr, cat_obs_ptr, preds_ptr, n_samples, n_num_features, n_cat_features, start_tree_idx, stop_tree_idx);  
    }, py::arg("obs"), py::arg("categorical_obs"), py::arg("start_preds"), py::arg("start_tree_idx")=0, py::arg("stop_tree_idx")=0, "Predict using the model");
        // saveToFile method
    gbrl.def("save", [](GBRL &self, const std::string& filename) -> int {
        py::gil_scoped_release release; 
        return self.saveToFile(filename); 
    }, "Save the model to a file");
    gbrl.def("export", [](GBRL &self, const std::string& filename, const std::string& modelname) -> int {
        py::gil_scoped_release release; 
        return self.exportModel(filename, modelname); 
    }, py::arg("filename"), py::arg("modelname") = "", "Export model as a C-header file");
    gbrl.def("get_scheduler_lrs", [](GBRL &self) ->  py::array_t<float> {
        py::gil_scoped_release release; 
        float* lrs = self.get_scheduler_lrs(); 
        py::gil_scoped_acquire acquire;
        auto capsule = py::capsule(lrs, [](void* ptr) {
        delete[] reinterpret_cast<float*>(ptr);
        });
        return py::array({static_cast<long int>(self.opts.size())}, lrs, capsule);
    }, "Return current scheduler lrs");  
    gbrl.def("get_num_trees", [](GBRL &self) ->  int {
        py::gil_scoped_release release; 
        return self.get_num_trees(); 
    }, "Return current number of trees in the ensemble");  
    gbrl.def("get_metadata", [](GBRL &self) ->  py::dict {
        return metadataToDict(self.metadata); 
    }, "Return ensemble metadata");  
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
    }, "Print specified tree index");
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
    }, "Plot specified tree index to png file"); 
}