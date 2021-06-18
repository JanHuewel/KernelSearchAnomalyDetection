from Main import main
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari_score
import pickle
import subprocess

list_of_data = ["data/dd_test_basic_anomaly2.csv"]
list_of_segment_lengths = [100]
list_of_methods = ["cov"] #"likelihood", "MSE", "KLD", "sampling", "sampling2"]
list_of_clusterings = ["PIC", "Agg"]
list_of_normalizations = [0,1,2]
list_of_ground_truths = [[0,0,0,0,1,0,0,0,0,0]]

results_df = pd.DataFrame()

for i, data in enumerate(list_of_data):
    for method in list_of_methods:
        for clustering in list_of_clusterings:
            for normalization in list_of_normalizations:
                try:
                    result = main(data,
                                  segment_length=list_of_segment_lengths[i],
                                  method=method,
                                  clustering_method=clustering,
                                  normalization=normalization)
                    evaluation = ari_score(result, list_of_ground_truths[i])
                except:
                    evaluation = "ERROR"
                results_df = results_df.append({#'data': data,
                                   #'segment_length': list_of_segment_lengths[i],
                                   'method': method,
                                   'clustering_method': clustering,
                                   'normalization': normalization,
                                   'result': evaluation}, ignore_index=True)
print(results_df)
#pickle.dump(results_df, 'Results_06-18_pickle.txt')
#results_df.to_csv('Results_06-18.csv')