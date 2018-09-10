import pickle

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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
    file = 'NB-CV.pkl'
    # Load from file
    load_file = open(file,'rb')
    clf = pickle.load(load_file)
    print("Loading file : ",clf)
    datas, labels = LoadData("datas_stopword1.txt", "labels_new1.txt")
    # datas, labels = LoadData()
    print(len(datas))
    print(len(labels))
    X_train, X_valid, y_train, y_valid = train_test_split(datas, labels, test_size=0.2, random_state=50)
    vectorizer = CountVectorizer()  # Chuyển đổi định dạng text thành vector
    transformed_x_train = vectorizer.fit_transform(X_train).toarray()  # Chuyển X_train về dạng array
    # print(vectorizer.get_feature_names()) #Đó chính là các từ xuất hiện ít nhất 1 lần trong các String
    trainVocab = vectorizer.vocabulary_  # export tập từ vựng
    vectorizer = CountVectorizer(vocabulary=trainVocab)  # Chuyển X_valid về dạng Array
    transformed_x_valid = vectorizer.fit_transform(X_valid).toarray()
    y_pred1 = clf.predict(transformed_x_valid)
    print('Training size = %d, accuracy = %.2f%%' % \
          (len(X_train), accuracy_score(y_valid, y_pred1) * 100))

if __name__ == "__main__":
    Classification()


