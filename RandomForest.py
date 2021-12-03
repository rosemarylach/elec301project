import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


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

#run algorithm
x_train, x_test, y_train, y_test = train_test_split(params_norm, labels, test_size = 0.2, random_state=1)
mod = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy', max_features=15).fit(params_norm, labels)

y_predicted= mod.predict(x_test)

confusion_mat = confusion_matrix(y_test, y_predicted)
print(accuracy_score(y_predicted, y_test))

pred = mod.predict(test_params_norm)

sn.heatmap(confusion_mat, annot=True)
plt.xlabel("Random Forest Prediction")
plt.ylabel("Actual")
plt.show()

pred_frame = pd.DataFrame(columns=['filename', 'label'])
filenames = ['sample%03d.wav' % i for i in range(0,300)]
pred_frame = pd.DataFrame({'filename': filenames, 'label': pred})

# save features to csv
pred_frame.to_csv(r'C:\Users\elija\PycharmProjects\Music Classifier\mlr_class_pred5.csv', index=False)

