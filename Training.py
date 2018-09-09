import itertools
import json
from collections import Counter
from os.path import join

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np

pd.set_option('display.max_columns', 7)

def LoadData():
    datas = []
    labels = []
    with open("datas.txt",'r',encoding='utf-8')as file:
        for i in file:
            datas.append(i)

    with open("labels.txt",'r',encoding='utf-8')as file:
        for i in file:
            labels.append(i)
    return datas,labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def Classification():
    datas,labels = LoadData()
    print(len(datas))
    print(len(labels))
    X_train, X_valid, y_train, y_valid = train_test_split(datas, labels, test_size=0.2, random_state=50)#Chia file data thành 80% để train và 20% để test

    vectorizer = CountVectorizer()#Chuyển đổi định dạng text thành vector
    transformed_x_train = vectorizer.fit_transform(X_train).toarray()#Chuyển X_train về dạng array
    # print(vectorizer.get_feature_names()) #Đó chính là các từ xuất hiện ít nhất 1 lần trong các String
    trainVocab = vectorizer.vocabulary_ #export tập từ vựng
    vectorizer = CountVectorizer(vocabulary=trainVocab) #Chuyển X_valid về dạng Array
    transformed_x_valid = vectorizer.fit_transform(X_valid).toarray()

    best_clf = MultinomialNB()
    best_clf.fit(transformed_x_train, y_train)
    y_pred = best_clf.predict(transformed_x_valid)

    filename = 'NB-CV.pkl'
    saved_model = open(filename, 'wb')
    pickle.dump(best_clf, saved_model)
    saved_model.close()

    # print(np.asarray(y_valid))
    # print(np.asarray(y_pred))
    print('Training size = %d, accuracy = %.2f%%' % \
          (len(X_train), accuracy_score(y_valid, y_pred) * 100))
    print(np.linspace(0, 1, 10))
    params = {'alpha': np.linspace(0.1, 1, 10)}
    clf = MultinomialNB()
    clf = GridSearchCV(clf, params, cv=5)
    clf.fit(transformed_x_train, y_train)

    print(clf.best_params_)
    best_clf = clf.best_estimator_
    y_pred1 = best_clf.predict(transformed_x_valid)
    print('Training size = %d, accuracy = %.2f%%' % \
          (len(X_train), accuracy_score(y_valid, y_pred1) * 100))

    cnf_matrix = confusion_matrix(y_valid, y_pred1)
    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(cnf_matrix)

    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=labels,
    #                       title='Confusion matrix, without normalization')

    class_names=[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,156,188]

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    plt.savefig(join("images", "Confusion matrix, without normalization.png"))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(join("images", "Normalized confusion matrix.png"))

    plt.show()


if __name__ == "__main__":
    Classification()
