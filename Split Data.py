from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(params_norm, labels, test_size = 0.2, random_state=1)

y_predicted= mod.predict(x_test)
print(accuracy_score(y_predicted, y_test))