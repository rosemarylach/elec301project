# import libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

x_train, x_test, y_train, y_test = train_test_split(params_norm, labels, test_size = 0.25)
# x_train = params_norm
# x_test = test_params_norm
# y_train = labels

mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(100), max_iter=10000, learning_rate_init=0.01, random_state=1)
mlp.fit(x_train, y_train)
pred = mlp.predict(x_test)

print(accuracy_score(pred, y_test))

# pred_frame = pd.DataFrame(columns=['filename', 'label'])
# filenames = test_features.loc[:, 'filename'].values
# pred_frame = pd.DataFrame({'filename': filenames, 'label': pred})

# save features to csv
# pred_frame.to_csv('Predictions/mlp_results.csv', index=False)