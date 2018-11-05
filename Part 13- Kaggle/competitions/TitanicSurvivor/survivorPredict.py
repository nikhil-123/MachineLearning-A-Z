import survivorPredictDataPreprocessor as dp

X = dp.X_train
y = dp.y_train

#Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Creating classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
classifier.fit(X_train,y_train)

#Predicting the test results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#K-Cross validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)