import itertools
import pickle
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

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

    load_file = open(file,'rb')
    clf = pickle.load(load_file)
    print("Loading file : ",clf)

    X_train, y_train = LoadData("datas_stopword1.txt", "labels_new1.txt")
    X_valid, y_valid = LoadData("datas_valid.txt","labels_valid.txt")

    vectorizer = CountVectorizer()  # Chuyển đổi định dạng text thành vector
    transformed_x_train = vectorizer.fit_transform(X_train).toarray()  # Chuyển X_train về dạng array
    # print(vectorizer.get_feature_names()) #Đó chính là các từ xuất hiện ít nhất 1 lần trong các String
    trainVocab = vectorizer.vocabulary_  # export tập từ vựng
    vectorizer = CountVectorizer(vocabulary=trainVocab)  # Chuyển X_valid về dạng Array
    transformed_x_valid = vectorizer.fit_transform(X_valid).toarray()
    y_pred1 = clf.predict(transformed_x_valid)
    print('Training size = %d, accuracy = %.2f%%' % \
          (len(X_train), accuracy_score(y_valid, y_pred1) * 100))

    cnf_matrix = confusion_matrix(y_valid, y_pred1)
    np.set_printoptions(precision=2)
    print('Confusion matrix:')

    class_names = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 156, 188]

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


