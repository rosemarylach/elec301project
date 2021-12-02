import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# read features from csv
features = pd.read_csv('features.csv')
test_features = pd.read_csv('test_features.csv')

# create feature dataframe
feature_names = ["idx", "mean", "stdev", "skew", "kurtosis", "zcr_mean", "zcr_stdev",
                 "rmse_mean", "rmse_stdev", "tempo"] + \
                ['mfccs_' + str(i+1) + '_mean' for i in range(20)] + \
                ['mfccs_' + str(i+1) + '_stdev' for i in range(20)] + \
                ['chroma_' + str(i+1) + '_mean' for i in range(12)] + \
                ['chroma_' + str(i+1) + '_stdev' for i in range(12)] + \
                ["centroid_mean", "centroid_stdev"] + \
                ['contrast_' + str(i+1) + '_mean' for i in range(7)] + \
                ['contrast_' + str(i+1) + '_std' for i in range(7)] + \
                ["rolloff_mean", "rolloff_stdev", "genre"]

param_names = feature_names[1:-1]
label_names = feature_names[-1]

# extract parameters and labels
params = features.loc[:, param_names].values
labels = features.loc[:, label_names].values
test_params = test_features.loc[:, param_names].values

# normalize data
params_norm = StandardScaler().fit_transform(params)
test_params_norm = StandardScaler().fit_transform(test_params)

#generating and train model
model = KNeighborsClassifier(n_neighbors= 5)
model.fit(params_norm, labels)

#prediction
predict = model.predict(test_params_norm)

#output to csv
pred_frame = pd.DataFrame(columns=['filename', 'label'])
filenames = test_features.loc[:, 'filename'].values
pred_frame = pd.DataFrame({'filename': filenames, 'label': predict})

pred_frame.to_csv('knn_predictions1', encoding='utf-8', index=False)