import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# read features from csv
# features = pd.read_csv('features.csv')
# test_features = pd.read_csv('test_features.csv')
features = pd.read_csv('splitfeatures-train.csv')
test_features = pd.read_csv('splitfeatures-test.csv')

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
print(param_names)
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

mlr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=182).fit(x_train, y_train)
y_predicted= mlr.predict(x_test)
confusion_mat = confusion_matrix(y_test, y_predicted)

sn.heatmap(confusion_mat, annot=True)
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()


#print(accuracy_score(ypred_test, y_test))

pred = mlr.predict(test_params_norm)

pred_frame = pd.DataFrame(columns=['filename', 'label'])
filenames = test_features.loc[:, 'filename'].values
pred_frame = pd.DataFrame({'filename': filenames, 'label': pred})

# save features to csv
# pred_frame.to_csv(r'C:\Users\elija\PycharmProjects\Music Classifier\mlr_class_pred3.csv', index=False)
pred_frame.to_csv('Predictions/lr_pred.csv', index=False)
