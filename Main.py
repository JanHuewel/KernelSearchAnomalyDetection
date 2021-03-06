from itertools import product
import configparser
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
from PIC import pic
from sklearn.cluster import KMeans, AgglomerativeClustering
from gpbasics.Statistics.GaussianProcess import GaussianProcess
from sklearn.metrics import adjusted_rand_score as ari_score

DEBUG = False


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

def kernel_search(dataset_name, segment_length = 100):
    """
    Partition the data and perform segment-wise kernel search via CKS.
    Return segments, kernel expressions and noises.

    Parameters:
    dataset_name: string, name of the file containing the data
    segment_length: int, length of an individual segment
    """

    data_normalization = "Z-scale" # Z-scale / 0-1

    # check length of dataset
    dataset_pandas = pd.read_csv(dataset_name)
    dataset_length = len(dataset_pandas)

    algorithms = [agr.AlgorithmType.CKS]
    options = {"global_max_depth": 3}

    # prepare data
    data_x = dataset_pandas['X'].to_numpy().reshape((dataset_length, 1))
    data_y = dataset_pandas['Y'].to_numpy().reshape((dataset_length, 1))

    if data_normalization == "Z-scale":
        data_y -= data_y.mean()
        data_y /= data_y.std()
        data_x = data_x - data_x.mean()
        data_x = data_x / data_x.std()
    elif data_normalization == "0-1":
        data_y -= data_y.min()
        data_y /= data_y.max()
        data_x -= data_x.min()
        data_x = data_x / data_x.max()

    # create segments
    datasets = list()
    for i in range(int(dataset_length/segment_length)):
        data_input_format = di.DataInput(data_x[i*segment_length:(i+1)*segment_length],
                                         data_y[i*segment_length:(i+1)*segment_length],
                                         data_x[i*segment_length:(i+1)*segment_length],
                                         data_y[i*segment_length:(i+1)*segment_length])
        datasets.append(data_input_format)

    # perform kernel search on segments
    list_of_kernels = []
    list_of_noises = []
    for i in range(int(dataset_length/segment_length)):
        exps = exp.execute_retrieval_experiments_set(
            datasets[i], algorithms=algorithms, mean_function_search=False, options=options, illustrate=False,
            model_selection_metric=met.MetricType.LL, local_approx=mht.MatrixApproximations.NONE,
            numerical_matrix_handling=mht.NumericalMatrixHandlingType.CHOLESKY_BASED, optimize_metric=met.MetricType.LL,
            random_restart=10)

        list_of_kernels.append(exps[1]["best_gp"].covariance_matrix.kernel)
        list_of_noises.append(exps[1]["best_gp"].covariance_matrix.kernel.get_noise())

    return datasets, list_of_kernels, list_of_noises

def get_clusters(dataset_name, datasets, list_of_kernels, list_of_noises, segment_length = 100, method = "cov", clustering_method = "PIC", number_of_clusters = 2,
         normalization = 0, visual_output = False, text_output = True, output_filename="output.txt"):
    """
    Take the results of kernel_search and create clusters based on the additional parameters.

    Parameters:
    dataset_name: string, name of the file containing the dataset
    datasets: list(DataInput), list of segments
    list_of_kernels: list(Kernel), list of kernels
    list_of_noises: list(tf.Tensor), list of jitters for the segments
    segment_length: int, length of individual segments
    method: string ('cov', 'likelihood', 'MSE', 'KLD', 'sampling', 'sampling2'), method used to build affinity/distance matrix
    clustering_method: string ('PIC', 'Agg'), method used for clustering. PIC uses affinity matrices, Agglomerative clustering uses distances
    number_of_clusters: int, amount of clusters
    normalization: int (0,1,2), method of normalization. For the paper we only used 0 and 1
    visual_output: bool, determines if plots are shown
    text_output: bool, determines if text is displayed / written to a file
    output_filename: string, name of file that is used to save outputs
    """
    # Number of samples drawn in the sampling method
    number_of_samples = 500
    # check length of dataset
    dataset_pandas = pd.read_csv(dataset_name)
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
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j,i] = 1.0 / (tf.norm(K1 - K2, axis=[-1,-2]).numpy())
                else:
                    results_matrix[i, j] = results_matrix[j,i] = tf.norm(K1 - K2, axis=[-1,-2]).numpy()

    elif method == "likelihood":
        # build matrix out of likelihoods
        def loglike(covariance : tf.Tensor, noise : tf.Tensor, data : tf.Tensor):
            K2 : tf.Tensor = covariance + noise.numpy() * tf.eye(len(data), dtype=tf.float64)
            L = tf.linalg.cholesky(K2)
            alpha = tf.linalg.cholesky_solve(L, data)
            return -0.5 * tf.transpose(data) @ alpha - sum([tf.math.log(L[i,i]) for i in range(len(data))]) - 0.5 * len(data) * tf.cast(tf.math.log(2 * np.pi), tf.float64)

        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i+1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(datasets[j])
                cov_matrix_j.set_data_input(datasets[i])
                a = loglike(cov_matrix_i.get_K(list_of_kernels[i].get_last_hyper_parameter()), list_of_noises[i], datasets[j].data_y_train)
                b = loglike(cov_matrix_j.get_K(list_of_kernels[j].get_last_hyper_parameter()), list_of_noises[j], datasets[i].data_y_train)
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j, i] = -(a + b).numpy()
                else:
                    results_matrix[i, j] = results_matrix[j, i] = 1.0 / (- (a + b).numpy())


    elif method == "MSE":
        # build matrix with Mean Square Error
        # the MSE is computed between posterior predictions of data and actual target data in the segments
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i + 1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(di.DataInput(datasets[j].data_x_train,datasets[j].data_y_train, datasets[i].data_x_train, datasets[i].data_y_train))
                cov_matrix_j.set_data_input(di.DataInput(datasets[i].data_x_train,datasets[i].data_y_train, datasets[j].data_x_train, datasets[j].data_y_train))
                K_s_i = cov_matrix_i.get_K_s(list_of_kernels[i].get_last_hyper_parameter())
                K_s_j = cov_matrix_j.get_K_s(list_of_kernels[j].get_last_hyper_parameter())
                cov_matrix_i.set_data_input(di.DataInput(datasets[i].data_x_train, datasets[i].data_y_train, datasets[j].data_x_train,datasets[j].data_y_train))
                cov_matrix_j.set_data_input(di.DataInput(datasets[j].data_x_train, datasets[j].data_y_train, datasets[i].data_x_train,datasets[i].data_y_train))
                prediction_j = K_s_i \
                          @ cov_matrix_i.get_K_inv(list_of_kernels[i].get_last_hyper_parameter(), list_of_noises[i]) \
                          @ datasets[i].data_y_train
                prediction_i = K_s_j \
                          @ cov_matrix_j.get_K_inv(list_of_kernels[j].get_last_hyper_parameter(), list_of_noises[j]) \
                          @ datasets[j].data_y_train
                error_i = sum(tf.math.square(prediction_i - datasets[i].data_y_train)) / len(prediction_i)
                error_j = sum(tf.math.square(prediction_j - datasets[j].data_y_train)) / len(prediction_j)
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j, i] = 1.0 / (error_i + error_j)
                else:
                    results_matrix[i, j] = results_matrix[j, i] = (error_i + error_j)

    elif method == "KLD":
        # build matrix based on Kullback Leibler Divergence
        # for each pair of segments, determine the divergence between multivariate normal distributions with zero mean
        # and covariance equal to that induced by the kernels.
        # only applicable of total length of dataset is dividable by segment_length
        def kld(sigma0: tf.Tensor, sigma1: tf.Tensor):
            L0 = tf.linalg.cholesky(sigma0)
            L1 = tf.linalg.cholesky(sigma1)
            return 0.5 * (tf.linalg.trace(tf.linalg.inv(sigma1) @ sigma0) + 0.0 - segment_length \
                          + sum([tf.math.log(L1[i,i]) for i in range(tf.shape(L1)[0])])\
                          - sum([tf.math.log(L0[i,i]) for i in range(tf.shape(L1)[0])]))
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            cov_matrix_i = cov.HolisticCovarianceMatrix(list_of_kernels[i])
            for j in range(i + 1):
                cov_matrix_j = cov.HolisticCovarianceMatrix(list_of_kernels[j])
                cov_matrix_i.set_data_input(datasets[i])
                cov_matrix_j.set_data_input(datasets[j])
                K1 = cov_matrix_i.get_K_noised(list_of_kernels[i].get_last_hyper_parameter(), list_of_noises[i])
                K2 = cov_matrix_j.get_K_noised(list_of_kernels[j].get_last_hyper_parameter(), list_of_noises[j])
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j, i] = 1.0 / (kld(K1, K2) + kld(K2, K1) + 1.0)
                else:
                    results_matrix[i, j] = results_matrix[j, i] = kld(K1, K2) + kld(K2, K1)
                if DEBUG:
                    # ----
                    if i == 2 and j == 1:
                        print("covariance matrix")
                        print(f"{i}, {j} \nK1 : {np.round(K1, 3)} \nK2 : {np.round(K2, 3)}")
                        print("eigenvalues")
                        print(f"K1: \n {tf.linalg.eigvals(K1)}\n K2: \n {tf.linalg.eigvals(K2)}")
                        print(f"Determinante: \n K1: {tf.linalg.det(K1)}, K2: {tf.linalg.det(K2)}")
                        print("are diagonal entries maximal?")
                        print(
                            f"K1: {all([[(K1[i, i] > K1[i, j] or i == j) for j in range(np.shape(K1)[1])] for i in range(np.shape(K1)[0])])}")
                        print(
                            f"K2: {all([[(K2[i, i] > K2[i, j] or i == j) for j in range(np.shape(K2)[1])] for i in range(np.shape(K2)[0])])}")
                        print(f"term: {tf.linalg.trace(tf.linalg.inv(K1) @ K2)}")
                    # ----

    elif method == "sampling":
        # build matrix based on posterior samples of the GPs
        # Given two segments, draw samples from their respective GPs over a comparatively large sample interval and
        # determine the difference between the samples
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            for j in range(i + 1):
                gp_i = GaussianProcess(list_of_kernels[i], bmf.ZeroMeanFunction(1))
                gp_j = GaussianProcess(list_of_kernels[j], bmf.ZeroMeanFunction(1))
                data_for_gp = di.DataInput(datasets[i].join(datasets[j]).data_x_train, datasets[i].join(datasets[j]).data_y_train,
                                           tf.reshape(tf.linspace(-5,5,100), [100,1]), tf.reshape(tf.linspace(0,0,100),[100,1]))
                data_for_gp.set_mean_function(bmf.ZeroMeanFunction(1))
                gp_i.set_data_input(data_for_gp)
                gp_j.set_data_input(data_for_gp)
                prediction_i = gp_i.get_n_posterior_functions(number_of_samples, list_of_kernels[i].get_last_hyper_parameter(), list_of_noises[i])
                prediction_j = gp_j.get_n_posterior_functions(number_of_samples, list_of_kernels[j].get_last_hyper_parameter(), list_of_noises[j])
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j, i] = number_of_samples / sum(tf.math.reduce_max(abs(prediction_i - prediction_j), axis = 0))
                else:
                    results_matrix[i, j] = results_matrix[j, i] = sum(tf.math.reduce_max(abs(prediction_i - prediction_j), axis=0)) / number_of_samples

    elif method == "sampling2":
        # build matrix based on prior samples of the GPs
        # Given two segments, draw prior samples from their respective GPs over a comparatively large sample interval and
        # determine the difference between the samples
        results_matrix = np.zeros((len(datasets), len(datasets)))
        for i in range(len(datasets)):
            for j in range(i + 1):
                gp_i = GaussianProcess(list_of_kernels[i], bmf.ZeroMeanFunction(1))
                gp_j = GaussianProcess(list_of_kernels[j], bmf.ZeroMeanFunction(1))
                data_for_gp = di.DataInput(datasets[i].join(datasets[j]).data_x_train, datasets[i].join(datasets[j]).data_y_train,
                                           tf.reshape(tf.linspace(-5,5,100), [100,1]), tf.reshape(tf.linspace(0,0,100),[100,1]))
                data_for_gp.set_mean_function(bmf.ZeroMeanFunction(1))
                gp_i.set_data_input(data_for_gp)
                gp_j.set_data_input(data_for_gp)
                prediction_i = gp_i.get_n_prior_functions(number_of_samples, list_of_kernels[i].get_last_hyper_parameter(), list_of_noises[i])
                prediction_j = gp_j.get_n_prior_functions(number_of_samples, list_of_kernels[j].get_last_hyper_parameter(), list_of_noises[j])
                if clustering_method == "PIC":
                    results_matrix[i, j] = results_matrix[j, i] = number_of_samples / sum(tf.math.reduce_max(abs(prediction_i - prediction_j), axis=0))
                else:
                    results_matrix[i, j] = results_matrix[j, i] = sum(
                        tf.math.reduce_max(abs(prediction_i - prediction_j), axis=0)) / number_of_samples

    np.set_printoptions(threshold=np.inf)
    # norm results
    output = {}
    if normalization and text_output:
        output["pre_norm"] = np.round(results_matrix, 3).tolist()
    if normalization == 1: # shift matrix to be all non-negative. scale diagonal to 1, then set it to 0
        results_matrix -= min(0, results_matrix.min())
        for i in range(len(datasets)):
            results_matrix[i, :] /= np.sqrt(results_matrix[i, i])
        for i in range(len(datasets)):
            results_matrix[:, i] /= results_matrix[i, i]
            results_matrix[i, i] = 0
    elif normalization == 2: # diagonal entries must be positive. scale diagonal to 1
        for i in range(len(datasets)):
            results_matrix[i, :] /= np.sqrt(results_matrix[i, i])
        for i in range(len(datasets)):
            results_matrix[:, i] /= results_matrix[i, i]
            results_matrix[i, i] = 0

    # clustering
    if clustering_method == "PIC":
        x = pic(results_matrix, 1000, 1e-6)
        clustering = KMeans(n_clusters = number_of_clusters).fit(x)
    else:
        clustering = AgglomerativeClustering(number_of_clusters, affinity="precomputed", linkage="complete").fit(results_matrix)

    # terminal output to check results
    if text_output:
        if DEBUG:
            print(f"results: \n{np.round(results_matrix, 2)}")
            print(f"PIC x: \n{x}")
            print("list of kernels:")
            for i, kernel in enumerate(list_of_kernels):
                print(f"{i}: {kernel.get_string_representation()}, {[entry.numpy() for entry in kernel.get_last_hyper_parameter()]}, noise: {kernel.noise}")
            print(f"labels: \n{clustering.labels_}")
        output["results"] = np.round(results_matrix, 2).tolist()
        if clustering_method == "PIC":
            output["PIC"] = x.tolist()
        list_of_kernels_output = {}
        for i, kernel in enumerate(list_of_kernels):
            list_of_kernels_output[f"{i}"] = {"string": kernel.get_string_representation(),
                                              "hyper": [entry.numpy().tolist() for entry in kernel.get_last_hyper_parameter()],
                                              "noise": np.float64(kernel.noise)
                                              }
        output["kernels"] = list_of_kernels_output
        output["labels"] = clustering.labels_.tolist()
        with open(f"{output_filename[:-4]}.json", "w") as write_file:
            json.dump(output, write_file, indent=4)


    # plot results
    if visual_output:
        fig,ax = plt.subplots()
        ax.scatter(x = dataset_pandas['X'], y = dataset_pandas['Y'], color='blue', alpha=0.2, marker='.')
        ax.set_xlabel('Data x')
        ax.set_ylabel('Data y')
        color_palet = ['red', 'blue', 'green', 'yellow', 'pink', 'brown', 'cyan', 'darkcyan', 'darkviolet', 'royalblue', 'tan', 'lightgreen', 'lime']
        for i in range(len(datasets)):
            #pdb.set_trace()
            x_lim_min = dataset_pandas['X'][i*segment_length]
            x_lim_max = dataset_pandas['X'][(i+1)*segment_length-1]
            #print(f"{x_lim_min} - {x_lim_max}")
            ax.axvspan(x_lim_min, x_lim_max, facecolor=color_palet[clustering.labels_[i]], alpha=0.4)
        plt.savefig(f'{output_filename[:-4]}.png', bbox_inches='tight')
        #plt.show()

    return clustering.labels_

def run_cluster_search_and_store(params):
    """
    Perform kernel search and clustering based on the parameters in params.
    Compare the results to the ground truth and save all outputs and the error to a file.

    Parameters:
    params: list, contains dataset, segment_length, datasets, list_of_kernels, list_of_noises, config
    """
    dataset, segment_length, datasets, list_of_kernels, list_of_noises, config = params
    data_split = dataset.split("/")
    if len(data_split) == 1:
        output_path = "Results/" + dataset + "_" + segment_length + "_" + "_".join(config)
    else:
        output_path = "Results/" + str(data_split[-1][:-4]) + "_" + segment_length + "_" + "_".join(config) + "_result.txt"
    try:
       labels = get_clusters(dataset_name=dataset,
                 datasets=datasets,
                 list_of_kernels=list_of_kernels,
                 list_of_noises=list_of_noises,
                 segment_length=int(segment_length),
                 method=config[0],
                 clustering_method=config[1],
                 normalization=int(config[2]),
                 visual_output=True,
                 output_filename=output_path)
    except Exception as e:
        print(e)
        labels = "ERROR"

    ground_truth_df= pd.read_csv(dataset)
    ground_truth = ground_truth_df["Anomaly"]
    ground_truth_labels = []
    for i in range(len(datasets)):
        block = list(ground_truth[i*int(segment_length):(i+1)*int(segment_length)])
        # ground_truth_labels.append(max(set(block), key=block.count)) # select if is majority of segment (useful in multiclass)
        ground_truth_labels.append(max(block)) # select if exists in segment (if only 2 categories exist)
    if not labels == "ERROR":
        result = ari_score(labels, ground_truth_labels)
    else:
        result = "ERROR"
    if os.path.exists(f"{output_path[:-4]}.json"):
        with open(f"{output_path[:-4]}.json", "r") as read_file:
            temp = json.load(read_file)
            temp["ARI"] = result
            temp["ground_truth"] = ground_truth_labels
        with open(f"{output_path[:-4]}.json", "w") as read_file:
            json.dump(temp, read_file, indent=3)



def main():
    """
    Read parameters from sys.argv if provided, otherwise from config.ini file.
    Perform kernel search and clustering based on provided parameters.
    Compare results to ground truth and save the error additionally to the normal output to a file.
    """
    if len(sys.argv) == 6:
        # if provided, use sys.argv as parameters
        datasets, list_of_kernels, list_of_noises = kernel_search(sys.argv[1],int(sys.argv[2]))
        labels = get_clusters(dataset_name=sys.argv[1],
                             datasets=datasets,
                             list_of_kernels=list_of_kernels,
                             list_of_noises=list_of_noises,
                             segment_length=int(sys.argv[2]),
                             method=sys.argv[3],
                             clustering_method=sys.argv[4],
                             normalization=int(sys.argv[5]),
                             visual_output=False,
                             text_output=False)
        ground_truth_df = pd.read_csv(sys.argv[1])
        ground_truth = ground_truth_df["Anomaly"]
        ground_truth_labels = []
        for i in range(len(datasets)):
            block = list(ground_truth[i*int(sys.argv[2]):i+1*int(sys.argv[2])])
            #ground_truth_labels.append(max(set(block), key=block.count))
            ground_truth_labels.append(max(block))
        result = ari_score(labels, ground_truth_labels)
        results_file = open("Results/" + sys.argv[1][:4] +  "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_result.txt", "a")
        results_file.write(result)
        results_file.close()
    else:
        # if sys.argv is not sufficiently given, get parameters from config.ini and pool over all given combinations
        config = configparser.ConfigParser()
        config.read("config.ini")
        dataset_names = config["dataset_names"]["dataset_names"].split(',')
        segment_lengths = config["segment_lengths"]["segment_lengths"].split(',')
        metrics = config["metrics"]["metrics"].split(',')
        clustering_methods = config["clustering_methods"]["clustering_methods"].split(',')
        normalization_methods = config["normalization_methods"]["normalization_methods"].split(',')

        configs = list(product(metrics, clustering_methods, normalization_methods))
        kernel_search_combinations = list(product(dataset_names, segment_lengths))
        from multiprocessing import Pool

        for dataset, segment_length in kernel_search_combinations:
            datasets, list_of_kernels, list_of_noises = kernel_search(dataset, int(segment_length))
            total_combinations = [[dataset, segment_length, datasets, list_of_kernels, list_of_noises, config] for config in configs]
            with Pool(len(total_combinations)) as p:
                p.map(run_cluster_search_and_store, total_combinations)

            # Load config file
            # Iterate over all combinations of the configs
            # Store the configurations in a separate file
            # Remove the line of the configurations that you have already executed

if __name__=="__main__":
    main()
