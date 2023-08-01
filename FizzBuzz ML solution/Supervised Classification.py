import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from Generate_Dataset_ML_Fizz_Buzz import Dataset_Generator_ML_Fizz_Buzz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.externals
import joblib
import coverage

cov = coverage.Coverage()
cov.start()


# Generate Dataset X in binary representation and y labels ()
length_data = 1024
num_digits = 10
# Encoder the number in binary representation with num_digits length
X, y = Dataset_Generator_ML_Fizz_Buzz(length_data, num_digits)
# Encoder the labels ('Fizz','FizzBuzz', 'Buzz', 'None') in (0,1,2,3)
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y.T)
# Count the number of samples for each class, imbalanced problem


# Dividing X, y into training and test data, we use the first 100 number for test set
X_train = X[100:]
y_train = y[100:]
X_test = X[:100]
y_test = y[:100]



# Random permutation in train set
X_train, y_train= shuffle(X_train, y_train, random_state = 42)

# Preprocessing the data
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Logistic Regression Model for multi-class classification, l2-regularization
softmax_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg")
softmax_reg.fit(X, y)
score = softmax_reg.score(X_test, y_test)
print("Accuracy_LR_softmax:", score)


# Training a KNN classifier
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
#print("Accuracy_KNN:", accuracy)


# Training a RF classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy_RF:", accuracy)


# Training a SVM classifier
svc = svm.SVC(kernel='linear', C=1.0, random_state=1) # Linear Kernel
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
#print("Accuracy_SVM_linear:", accuracy_score(y_test, y_pred))


# SGDClassifier for unbalanced data
sgdc = SGDClassifier(loss='hinge', max_iter=1000, tol=0.01)
sgdc.fit(X_train, y_train)
score = sgdc.score(X_train, y_train)
y_pred = sgdc.predict(X_test)
#print("Accuracy_SDGC:", score)


# Cross_validation for each Model
from sklearn.model_selection import KFold

scores = np.zeros((5,10))

kFold = KFold(n_splits=10, random_state=42, shuffle=True)

split=0

for train_index, test_index in kFold.split(X):

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    softmax_reg.fit(X_train, y_train)
    accuracy = softmax_reg.score(X_test, y_test)
    scores[0,split] = accuracy

    knn=KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    scores[1,split] = accuracy

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores[2,split] = accuracy

    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    scores[3,split] = accuracy

    sgdc.fit(X_train, y_train)
    accuracy = sgdc.score(X_test, y_test)
    scores[4,split] = accuracy

    split =split + 1

model = ['LR', 'KNN', 'RF', 'SVM', 'SGDC']
means = []
for i in range(0,len(model)):
    means.append(np.mean(scores[i,:]))

print("The best accuracy model is: " + model[means.index(max(means))] + " score = ", max(means))


cov.stop()
cov.save()

cov.html_report()
