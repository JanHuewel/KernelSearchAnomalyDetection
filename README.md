This project is the source code belonging to the paper "On Kernel Search Based Gaussian Process Anomaly Detection" and was used to run the experiments there as well as train the used models.

At the time of writing the dataset did not provide ground truth information, thus we used different GT labels.

Our labels for the files were as follows:

| Dataset | Start | End | 
| 4 | 5400 | 5700 | 
| 8 | 5550 | 5600 |
| 9 | 4800 | 4900 | 
| 22 | 6500 | 6700 | 
| 24 | 4450 | 4600 | 
| 25 | 5500 | 5700 | 

Since our experiments, the GT labels have been published with the dataset at [the website of Wu and Keogh](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/).

Our labels are very close the ground truth given by Wu and Keogh (i.e. our labels are enclosed in their boundaries), which warrants accuracy for our results.


