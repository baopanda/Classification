import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np

pd.set_option('display.max_columns', 7)

def LoadData(path_data,path_label):
    datas = []
    labels = []
    with open(path_data, 'r', encoding='utf-8')as file:
        for i in file:
            datas.append(i)

    with open(path_label, 'r', encoding='utf-8')as file:
        for i in file:
            labels.append(i)
    return datas, labels

def Classification():
    X_train, y_train = LoadData("datas_stopword1.txt", "labels_new1.txt")
    X_valid, y_valid = LoadData("datas_valid.txt", "labels_valid.txt")
    # X_train, X_valid, y_train, y_valid = train_test_split(datas, labels, test_size=0.2, random_state=50)#Chia file data thành 80% để train và 20% để test

    vectorizer = CountVectorizer()#Chuyển đổi định dạng text thành vector
    transformed_x_train = vectorizer.fit_transform(X_train).toarray()#Chuyển X_train về dạng array
    # print(vectorizer.get_feature_names()) #Đó chính là các từ xuất hiện ít nhất 1 lần trong các String
    trainVocab = vectorizer.vocabulary_ #export tập từ vựng
    vectorizer = CountVectorizer(vocabulary=trainVocab) #Chuyển X_valid về dạng Array
    transformed_x_valid = vectorizer.fit_transform(X_valid).toarray()

    best_clf = MultinomialNB()
    best_clf.fit(transformed_x_train, y_train)
    y_pred = best_clf.predict(transformed_x_valid)

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

    filename = 'NB-CV.pkl'
    saved_model = open(filename, 'wb')
    pickle.dump(best_clf, saved_model)
    saved_model.close()

if __name__ == "__main__":
    Classification()
