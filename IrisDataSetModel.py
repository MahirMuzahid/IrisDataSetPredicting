#Support Vector Machine

#importing modules
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

#load dataset
iris = datasets.load_iris()

#load dependent and independent value
x = iris.data[:, 3]
y = iris.target

#split ore data for test and trai
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=4)

#reshape data in 1D
X_train_mod = X_train.reshape(-1, 1)
X_test_mod = X_test.reshape(-1, 1)
Y_train_mod = Y_train.reshape(-1, 1)
Y_test_mod = Y_test.reshape(-1, 1)

#load SVM algorithm
model = svm.SVC(kernel = 'linear')

#fit model or train model
model.fit(X_train_mod, Y_train_mod)

#load predicted value for test data
Y_pred_mod = model.predict(X_test_mod)

#getting accurecy
print(metrics.r2_score(Y_test_mod, Y_pred_mod))

