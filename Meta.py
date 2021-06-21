import Main
from Main import kernel_search, get_clusters
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari_score
import pickle
import multiprocessing

list_of_data = ["data/dd_test_basic_anomaly2.csv"]
list_of_segment_lengths = [50]
list_of_methods = ["sampling", "sampling2"]
list_of_clusterings = ["PIC", "Agg"]
list_of_normalizations = [0,1,2]
list_of_ground_truths = [[0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]]#[[0,0,0,0,1,0,0,0,0,0]]

def main():
    results_df = pd.DataFrame()
    for i, data in enumerate(list_of_data):
        datasets, list_of_kernels, list_of_noises = kernel_search(data, list_of_segment_lengths[i])
        for method in list_of_methods:
            for clustering in list_of_clusterings:
                for normalization in list_of_normalizations:
                    try:
                        labels = get_clusters(dataset_name=data,
                                              datasets=datasets,
                                              list_of_kernels=list_of_kernels,
                                              list_of_noises=list_of_noises,
                                              segment_length=list_of_segment_lengths[i],
                                              method=method,
                                              clustering_method=clustering,
                                              normalization=normalization,
                                              visual_output=False,
                                              text_output=False)
                        evaluation = ari_score(labels, list_of_ground_truths[i])
                    except Exception as E:
                        print(E)
                        evaluation = "ERROR"
                    results_df = results_df.append({#'data': data,
                                       #'segment_length': list_of_segment_lengths[i],
                                       'method': method,
                                       'clustering_method': clustering,
                                       'normalization': normalization,
                                       'result': evaluation}, ignore_index=True)
    print(results_df)
    #pickle.dump(results_df, 'Results_06-18_pickle.txt')
    results_df.to_csv('Results_06-18.csv')

if __name__=="__main__":
    multiprocessing.freeze_support()
    main()