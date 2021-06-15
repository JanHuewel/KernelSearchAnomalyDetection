import multiprocessing
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

# This file has to be put directly in folder "main/"
#   all other dependent packages need to be in sub folders of "main/"
if os.name != 'nt':
    sys.path.append(os.getcwd())

import gpbasics.global_parameters as global_param

global_param.init(tf_parallel=os.cpu_count())

import gpbasics.Statistics.CovarianceMatrix as cov
import gpbasics.DataHandling.DatasetHandler as dsh
import gpbasics.DataHandling.DataInput as di
import gpminference.Experiments.Experiment as exp
import gpminference.AutomaticGpmRetrieval as agr
import gpbasics.KernelBasics.BaseKernels as bk
import gpbasics.MeanFunctionBasics.BaseMeanFunctions as bmf
import gpbasics.Metrics.Metrics as met
import gpminference.KernelExpansionStrategies.KernelExpansionStrategy as kexp
import gpbasics.Optimizer.Fitter as f
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpminference.autogpmr_parameters as auto_gpm_param
import tensorflow as tf
import numpy as np
import logging
from PIC import pic
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import random

global_param.p_max_threads = os.cpu_count()
global_param.p_used_base_kernel = [bk.PeriodicKernel,
                                   bk.SquaredExponentialKernel,
                                   bk.LinearKernel]

global_param.p_used_base_mean_functions = [bmf.ConstantMeanFunction, bmf.LinearMeanFunction]

global_param.p_default_hierarchical_kernel_expansion = \
    kexp.KernelExpansionStrategyType.BasicHierarchical

global_param.p_gradient_fitter = f.VariationalSgdFitter

auto_gpm_param.p_model_selection_with_test_data = True

global_param.p_dtype = tf.float64

global_param.p_cov_matrix_jitter = tf.constant(1e-8, dtype=global_param.p_dtype)

if __name__ == '__main__':
    dataset_name = "data/dd_test_basic_anomaly0.csv"
    segment_length = 5
    number_of_clusters = 2
    method = "KLD" # cov, likelihood, MSE, KLD
    normalization = False

    # check length of dataset
    dataset_pandas = pd.read_csv(dataset_name)
    dataset_length = len(dataset_pandas)

    algorithms = [
        # agr.AlgorithmType.CKS,
        # agr.AlgorithmType.ABCD,
         agr.AlgorithmType.SKC,
        # agr.AlgorithmType.SKS,  # 3CS
        # agr.AlgorithmType.IKS, # LARGe
        # agr.AlgorithmType.TopDown_HKS # LGI
    ]
    options = {"global_max_depth": 1, "local_max_depth": 3}

    # prepare data
    dataset = dsh.GeneralDatasetHandler(dataset_name,
                                        y_col_name='Y',
                                        x_col_name="X")
    #dataset = dsh.KDDHandler(2)
    datasets = list()
    a, b, c, d = dataset.get_splitted_data()
    for i in range(int(dataset_length/segment_length)):
        data_input_format = di.DataInput(a[i*segment_length:(i+1)*segment_length],
                                         b[i*segment_length:(i+1)*segment_length],
                                         c[i*segment_length:(i+1)*segment_length],
                                         d[i*segment_length:(i+1)*segment_length])
        datasets.append(data_input_format)

    # perform kernel search on segments
    list_of_kernels = []
    list_of_noises = []
    for i in range(int(dataset_length/segment_length)):

        exps = exp.execute_retrieval_experiments_set(
            datasets[i], algorithms=algorithms, mean_function_search=False, options=options, illustrate=False,
            model_selection_metric=met.MetricType.LL, local_approx=mht.MatrixApproximations.NONE,
            numerical_matrix_handling=mht.NumericalMatrixHandlingType.CHOLESKY_BASED, optimize_metric=met.MetricType.LL,
            random_restart=10) #to optimize

        list_of_kernels.append(exps[1]["best_gp"].covariance_matrix.kernel)
        list_of_noises.append(exps[1]["best_gp"].covariance_matrix.kernel.get_noise())

    # build distance matrix
    if method == "cov":
        # build matrix out of distances of cov matrices
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i+1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                data_for_cov_matrix = datasets[i].join(datasets[j])
                cov_matrix_i.set_data_input(data_for_cov_matrix)
                cov_matrix_j.set_data_input(data_for_cov_matrix)
                K1 = cov_matrix_i.get_K(list_of_kernels[i].get_last_hyper_parameter())
                K2 = cov_matrix_j.get_K(list_of_kernels[j].get_last_hyper_parameter())
                print(f"{i}, {j} \nK1 : {np.round(K1,1)} \nK2 : {np.round(K2,1)}")
                results_matrix[i,j] = results_matrix[j,i] = tf.norm(K1 - K2, axis=[-1,-2]).numpy() + abs(i-j) * 0.0

    elif method == "likelihood":
        # build matrix out of likelihoods
        def loglike(covariance : tf.Tensor, noise : tf.Tensor, data : tf.Tensor):
            K2 : tf.Tensor = covariance + noise.numpy() * tf.eye(len(data), dtype=tf.float64)
            #return -0.5 * tf.transpose(data) @ tf.linalg.inv(K2) @ data - 0.5 * tf.math.log(tf.linalg.det(K2)) - len(data)/2 * tf.cast(tf.math.log(2 * np.pi), tf.float64)
            L = tf.linalg.cholesky(K2)
            alpha = tf.linalg.cholesky_solve(L, data)
            return -0.5 * tf.transpose(data) @ alpha - sum([tf.math.log(L[i,i]) for i in range(len(data))]) - 0.5 * len(data) * tf.cast(tf.math.log(2 * np.pi), tf.float64)

        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i+1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(datasets[i])
                cov_matrix_j.set_data_input(datasets[j])
                a = loglike(cov_matrix_i.get_K(list_of_kernels[i].get_last_hyper_parameter()), list_of_noises[i], datasets[j].data_y_train)
                b = loglike(cov_matrix_j.get_K(list_of_kernels[j].get_last_hyper_parameter()), list_of_noises[j], datasets[i].data_y_train)
                results_matrix[i, j] = results_matrix[j, i] = (a + b).numpy()


    elif method == "MSE":
        # build matrix with MSE
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i + 1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(di.DataInput(datasets[i].data_x_train,datasets[i].data_y_train, datasets[j].data_x_train, datasets[j].data_y_train))
                cov_matrix_j.set_data_input(di.DataInput(datasets[j].data_x_train,datasets[j].data_y_train, datasets[i].data_x_train, datasets[i].data_y_train))
                prediction_j = cov_matrix_i.get_K_s(list_of_kernels[i].get_last_hyper_parameter()) \
                          @ cov_matrix_i.get_K_inv(list_of_kernels[i].get_last_hyper_parameter(), list_of_noises[i]) \
                          @ datasets[i].data_y_train
                prediction_i = cov_matrix_j.get_K_s(list_of_kernels[j].get_last_hyper_parameter()) \
                          @ cov_matrix_j.get_K_inv(list_of_kernels[j].get_last_hyper_parameter(), list_of_noises[j]) \
                          @ datasets[j].data_y_train
                error_i = sum(tf.math.square(prediction_i - datasets[i].data_y_train))
                error_j = sum(tf.math.square(prediction_j - datasets[j].data_y_train))
                results_matrix[i, j] = results_matrix[j, i] = error_i + error_j

    elif method == "KLD":
        def kld(sigma0: tf.Tensor, sigma1: tf.Tensor):
            return 0.5 * (tf.linalg.trace(tf.linalg.inv(sigma1) * sigma0) + 0.0 - len(datasets) + tf.math.log(tf.linalg.det(sigma1)/tf.linalg.det(sigma0)))
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i + 1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(datasets[i])
                cov_matrix_j.set_data_input(datasets[j])
                K1 = cov_matrix_i.get_K(list_of_kernels[i].get_last_hyper_parameter())
                K2 = cov_matrix_j.get_K(list_of_kernels[j].get_last_hyper_parameter())
                results_matrix[i, j] = results_matrix[j, i] = - kld(K1, K2) + kld(K2, K1)

    # norm results
    # results_matrix -= results_matrix.min()
    # results_matrix /= results_matrix.max()
    # ver 2
    if normalization:
        print(f"pre normalization results: {np.round(results_matrix, 3)}")
        results_matrix -= min(0, min([results_matrix[i, i] for i in range(len(datasets))]) - 1)
        for i in range(len(datasets)):
            results_matrix[i, :] /= np.sqrt(results_matrix[i, i])
        for i in range(len(datasets)):
            results_matrix[:, i] /= results_matrix[i, i]
            results_matrix[i, i] = 0

    # clustering
    x = pic(results_matrix, 1000, 1e-6)
    clustering = KMeans(n_clusters = number_of_clusters).fit(x)
    #clustering = DBSCAN(eps=0.00000001).fit(x)
    #clustering = AgglomerativeClustering(number_of_clusters, "precomputed", linkage="complete").fit(results_matrix)

    # terminal output to check results
    print(f"results: {np.round(results_matrix, 2)}")
    print(f"PIC x: {x}")
    for i, kernel in enumerate(list_of_kernels):
        print(f"{i}: {kernel.get_string_representation()}, {[entry.numpy() for entry in kernel.get_last_hyper_parameter()]}, noise: {kernel.noise}")

    # plot results
    fig,ax = plt.subplots()
    ax.scatter(x = dataset_pandas['X'], y = dataset_pandas['Y'], color='blue', alpha=0.2, marker='.')
    ax.set_xlabel('Data x')
    ax.set_ylabel('Data y')
    color_palet = ['red', 'blue', 'green', 'yellow', 'pink', 'brown', 'cyan', 'darkcyan', 'darkviolet', 'royalblue', 'tan', 'lightgreen', 'lime']
    for i in range(len(datasets)):
        x_lim_min = dataset_pandas['X'][i*segment_length]
        x_lim_max = dataset_pandas['X'][(i+1)*segment_length-1]
        #print(f"{x_lim_min} - {x_lim_max}")
        ax.axvspan(x_lim_min, x_lim_max, facecolor=color_palet[clustering.labels_[i]], alpha=0.4)
    plt.show()
