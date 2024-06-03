#include <random>
#include <chrono>
#include <string>

#include "gbrl.h"
#include "split_candidate_generator.h"
#include "optimizer.h"
#include "scheduler.h"
#include "loss.h"
#include "types.h"
#include "math_ops.h"

void print_mat(std::string name, const float *mat, const int size){
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i){
        std::cout << mat[i];
        if (i < size - 1)   
            std::cout << ", ";
    }
    std::cout << std::endl;
}


// int main() {
//     // Parameters
//     const int output_dim = 19;
//     const int policy_dim = 18;
//     const int max_depth = 4;
//     const int n_features = 128;
//     const int n_samples = 10000;
//     const int min_data_in_leaf = 1;
//     const int n_bins = 256;
//     const int par_th = 10; // Adjust as needed
//     const float cv_beta = 0.9;
//     // const scoreFunc split_score_func = L2; // Adjust the namespace as needed
//     const scoreFunc split_score_func = Cosine; // Adjust the namespace as needed
//     const generatorType generator_type = Quantile; // Adjust the namespace as needed
//     const bool use_control_variates = false;
//     const int batch_size = 10000;
//     const int verbose = 1;
//     const deviceType device = gpu;
//     int iters = 1;

//     const std::string load_model = "/swgwork/bfuhrer/projects/gbrl/saved_models/Krull-ramNoFrameskip-v4/gbrl_seed_0_1505280_steps.model";

//     // Random number generation
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::normal_distribution<> d(0, 1);

//     // Create a random Eigen matrix for obs
//     float *obs = new float[n_samples*n_features];
//     float *targets = new float[n_samples*output_dim];
//     float *grads = new float[n_samples*output_dim];
//     for (int i = 0; i < n_samples; ++i) {
//         for (int j = 0; j < n_features; ++j) {
//             obs[i*n_features + j] = d(gen);
//         }
//         for (int j = 0; j < output_dim; ++j) {
//             targets[i*output_dim + j] = d(gen);
//         }
//     }

//     GBRL model(output_dim, policy_dim, max_depth, min_data_in_leaf, n_bins, par_th, cv_beta, split_score_func, generator_type, use_control_variates, batch_size, verbose, device);
//     int status = model.loadFromFile(load_model);
//     if (status == 0){
//     int it = 0;
//         while (it < iters){
//             // Predict over obs to get the target predictions
//             float *target_predictions = model.predict(obs, n_samples, 0, model.get_num_trees());
//             set_zero_mat(grads, n_samples*output_dim, par_th);
//             // Calculate the loss and gradients
//             float loss = MultiRMSE::get_loss_and_gradients(target_predictions, targets, grads, n_samples, output_dim);
//             // Call the fit method
//             // print_mat("grads", grads, n_samples*output_dim);
//             model.fit(obs, grads, n_samples, n_features);
//             ++it;
//             // Print both losses
//             std::cout << it << ": Loss = " << loss << std::endl;
//             delete[] target_predictions;
//         }
//     }
    
//     // float *target_predictions = model.predict(obs, n_samples, 0, model.get_num_trees());
//     // delete[] target_predictions;

//     delete[] obs;
//     delete[] targets;
//     delete[] grads;
//     return 0;
// }

int main() {
    // Parameters
    const int output_dim = 10;
    const int policy_dim = 10;
    const int max_depth = 4;
    const int n_num_features = 50;
    const int n_cat_features = 0;
    const int n_samples = 10000;
    const int min_data_in_leaf = 1;
    const int n_bins = 256;
    const int par_th = 10; // Adjust as needed
    const float cv_beta = 0.9;
    // const scoreFunc split_score_func = L2; // Adjust the namespace as needed
    const scoreFunc split_score_func = Cosine; // Adjust the namespace as needed
    const generatorType generator_type = Quantile; // Adjust the namespace as needed
    const bool use_control_variates = false;
    const int batch_size = 10000;
    const int verbose = 1;
    const float lr = 0.01;
    const deviceType device = gpu;
    int iters = 1;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    // Create a random Eigen matrix for obs
    float *obs = new float[n_samples*n_num_features];
    char *categorical_obs = nullptr;

    float *targets = new float[n_samples*output_dim];
    float *grads = new float[n_samples*output_dim];
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_num_features; ++j) {
            obs[i*n_num_features + j] = d(gen);
        }
        for (int j = 0; j < output_dim; ++j) {
            targets[i*output_dim + j] = d(gen);
        }
    }
    // Calculate the mean of the target vector column-wise and set the model's bias
    float *target_mean =  calculate_mean(targets, n_samples, output_dim, par_th);
      // Instantiate GBRL

    GBRL model(output_dim, policy_dim, max_depth, min_data_in_leaf, n_bins, par_th, cv_beta, split_score_func, generator_type, use_control_variates, batch_size, verbose, device);
    model.set_bias(target_mean, output_dim);
    model.set_optimizer(SGD, Const, lr, 0.0, 2, 0.9, 0.999, 1.0e-8, 1.0e-5);
    // print_mat("obs", obs, n_samples*n_features);
    // print_mat("targets", targets, n_samples*output_dim);
    // print_mat("target_mean", target_mean, output_dim);
    int it = 0;
    while (it < iters){
        // Predict over obs to get the target predictions
        // loat* predict(const float *obs, const char *categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, int start_tree_idx, int stop_tree_idx);
        float *target_predictions = model.predict(obs, categorical_obs, n_samples, n_num_features, n_cat_features, 0, model.get_num_trees());
        set_zero_mat(grads, n_samples*output_dim, par_th);
        // Calculate the loss and gradients
        float loss = MultiRMSE::get_loss_and_gradients(target_predictions, targets, grads, n_samples, output_dim);
        // Call the fit method
        // print_mat("grads", grads, n_samples*output_dim);
        model.fit(obs, categorical_obs, grads, n_samples, n_num_features, n_cat_features);
        ++it;
        // Print both losses
        std::cout << it << ": Loss = " << loss << std::endl;
        delete[] target_predictions;
    }

    float *target_predictions = model.predict(obs, categorical_obs, n_samples, n_num_features, n_cat_features, 0, model.get_num_trees());
    delete[] target_predictions;
    // GBRL model2(output_dim, policy_dim, max_depth, min_data_in_leaf, n_bins, par_th, cv_beta, split_score_func, generator_type, use_control_variates, batch_size, verbose);
    // model2.set_optimizer(SGD, Const, lr, 0.0, 2, 0.9, 0.999, 1.0e-8, 1.0e-5);
    // // Perform fit with supervised learning for 100 iterations and print the final loss
    // float final_loss = model2.fit_sl(obs, targets, iters);
    // std::cout << "Final Loss after fit_sl: " << final_loss << std::endl;
    delete[] obs;
    delete[] targets;
    delete[] grads;
    delete[] target_mean;
    return 0;
}




// int main() {
//     const int n_samples = 100;
//     const int n_cols = 15;
//     const int par_th = 2;
//     Mat matrix(n_samples, n_cols);

//     // Fill the matrix with random values
//     std::default_random_engine generator;
//     std::uniform_real_distribution<float> distribution(0.0, 1.0);
//     for (int i = 0; i < n_samples; ++i) {
//         for (int j = 0; j < n_cols; ++j) {
//             matrix(i, j) = distribution(generator);
//         }
//     }

//     // Convert Eigen matrix to raw array for your functions
//     Mat different_mat = matrix;
//     float* mat = different_mat.data();
//     // Calculate mean and variance using your functions
//     auto start = std::chrono::high_resolution_clock::now();
//     float* mean = calculate_mean(mat, n_samples, n_cols, par_th);
//     auto stop = std::chrono::high_resolution_clock::now();
//     // Calculate the duration
//     auto mean_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//     float* var = calculate_var(mat, mean, n_samples, n_cols, par_th);
//     auto stop_var = std::chrono::high_resolution_clock::now();
//     auto var_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_var - start);

//     // Calculate mean and variance using Eigen
//     start = std::chrono::high_resolution_clock::now();
//     Vec eigen_mean = matrix.colwise().mean();
//     stop = std::chrono::high_resolution_clock::now();
//     auto eigen_mean_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//     Mat centered = matrix.rowwise() - eigen_mean.transpose();
//     Vec eigen_var = (centered.array().square().colwise().sum()) / (n_samples - 1);
//     stop_var = std::chrono::high_resolution_clock::now();
//     auto eigen_var_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_var - start);

//     print_mat("Mean", mean, n_cols);
//     std::cout << "eigen mean: " << eigen_mean.transpose() << std::endl;
//     print_mat("Var", var, n_cols);
//     std::cout << "eigen var: " << eigen_var.transpose() << std::endl;

//     std::cout << "time - mean: " << mean_duration.count() << " eigen mean: " << eigen_mean_duration.count() << " var duration: " << var_duration.count() << " eigen var duration: " << eigen_var_duration.count() << std::endl;

//     // Clean up
//     delete[] mean;
//     delete[] var;

//     // Convert Eigen matrix to raw array for your functions
//     Mat new_mat = matrix;
//     mat = new_mat.data();
//     // Step 1: Calculate mean using your function and Eigen
//     float* custom_mean = calculate_mean(mat, n_samples, n_cols, par_th);
//     eigen_mean = matrix.colwise().mean();

    
//     print_mat("Mean 2", custom_mean, n_cols);
//     std::cout << "eigen mean: " << eigen_mean.transpose() << std::endl;

//     // Step 2: Calculate variance using your function
//     float* custom_var = calculate_var_and_center(mat, custom_mean, n_samples, n_cols, par_th);

//     // Calculate variance using Eigen
//     centered = matrix.rowwise() - eigen_mean.transpose();
//     eigen_var = (centered.array().square().colwise().sum()) / (n_samples - 1);


//     // Compare variances
//     for (int i = 0; i < n_cols; ++i) {
//         if (std::abs(custom_var[i] - eigen_var[i]) > 1e-2) {
//             std::cerr << "Mismatch in variances at column " << i << std::endl;
//         }
//     }

//     // Step 3: Calculate standard deviation
//     for (int i = 0; i < n_cols; ++i) {
//         custom_var[i] = sqrtf(custom_var[i]);
//         eigen_var[i] = sqrtf(eigen_var[i]);
//     }

//     print_mat("Var2", custom_var, n_cols);
//     std:: cout << " Eigen var: " << eigen_var.transpose() << std::endl;


//     // Step 4: Standardize the matrix
//     // Using your functions
//     divide_mat_by_vec_inplace(mat, custom_var, n_samples, n_cols, par_th);


//     matrix = centered.array().rowwise() / (eigen_var.transpose().array() + 1e-8);

//     delete[] custom_mean;
//     delete[] custom_var;

//     float cv_beta = 0.9;
//     float num_trees = 2;

//     Mat new_mat2 = matrix;
//     mat = new_mat2.data();

//     Mat matrix_grad(n_samples, n_cols);

//     for (int i = 0; i < n_samples; ++i) {
//         for (int j = 0; j < n_cols; ++j) {
//             matrix_grad(i, j) = distribution(generator);
//         }
//     }

//         // Convert Eigen matrix to raw array for your functions
//     float* mat_grad = new float[n_samples * n_cols];
//     for (int i = 0; i < n_samples; ++i) {
//         for (int j = 0; j < n_cols; ++j) {
//             mat_grad[i * n_cols + j] = matrix_grad(i, j);
//         }
//     }



//     float error_correction = 1.0f / sqrtf(1.0f - powf(cv_beta, num_trees));
//     multiply_mat_by_scalar(mat, error_correction, n_samples, n_cols, par_th);

//     print_mat("after error correction", mat, n_samples*n_cols);

//     matrix.noalias() = (matrix.array() / sqrtf(1.0 - powf(cv_beta, num_trees))).matrix(); 

//     std::cout << "eigen after error correction: " << matrix << std::endl;

//     Mat centered_grads = matrix_grad.rowwise() - matrix_grad.colwise().mean();
//     Mat centered_momentum = matrix.rowwise() -  matrix.colwise().mean();

//     float* mean_grads = calculate_mean(mat_grad, n_samples, n_cols, par_th);
//     float* mean_momentum = calculate_mean(mat, n_samples, n_cols, par_th);

//     float *grads_copy = copy_mat(mat_grad, n_samples*n_cols, par_th);
//     float *variance_momentum = calculate_var_and_center(mat, mean_momentum, n_samples, n_cols, par_th);
//     subtract_vec_from_mat(grads_copy, mean_grads, n_samples, n_cols, par_th);

//     print_mat("centered grads ", grads_copy, n_samples*n_cols);
//     std::cout << "eigen centered grads: " << centered_grads << std::endl;

//     print_mat("centered momentum ", mat, n_samples*n_cols);
//     std::cout << "eigen centered momentum: " << centered_momentum << std::endl;

//     float *covariance = calculate_row_covariance(grads_copy, mat, n_samples, n_cols, par_th);


//     float* alpha_vec = element_wise_division(covariance, variance_momentum, n_cols, par_th);
//     for (int i = 0; i < n_cols; ++i){
//         if (alpha_vec[i] > 1 )
//             alpha_vec[i] = 1;
//         if (alpha_vec[i] < -1)
//             alpha_vec[i] = -1;
//     }
//     multiply_mat_by_vec_subtract_result(mat_grad, mat, alpha_vec, n_samples, n_cols, par_th);

    

//     Vec covariance_eigen = ((centered_grads.array() * centered_momentum.array()).colwise().sum() / (centered_grads.rows() - 1)).transpose();
//     Vec variance_eigen = (centered_momentum.array().square()).colwise().sum() / (centered_momentum.rows() - 1);
//     Vec alpha_eigen = covariance_eigen.array() / (variance_eigen.array() + 1e-8);
//     alpha_eigen = alpha_eigen.cwiseMax(-1).cwiseMin(1);
//     matrix_grad.noalias() -= (centered_momentum.array().rowwise() * alpha_eigen.transpose().array()).matrix();

//     print_mat("covariance ", covariance, n_cols);
//     std::cout << "eigen covariance: " << covariance_eigen.transpose() << std::endl;
//     print_mat("variance_momentum ", variance_momentum, n_cols);
//     std::cout << "variance_eigen: " << variance_eigen.transpose() << std::endl;
//     print_mat("alpha_vec ", alpha_vec, n_cols);
//     std::cout << "alpha_eigen: " << alpha_eigen.transpose() << std::endl;

//     print_mat("mat_grad ", mat_grad, n_samples*n_cols);
//     std::cout << "matrix_grad: " << matrix_grad << std::endl;

//     delete[] mat_grad;
//     delete[] covariance;
//     delete[] alpha_vec;
//     delete[] variance_momentum;
//     delete[] grads_copy;
//     delete[] mean_momentum;
//     delete[] mean_grads;


//     return 0;
// }