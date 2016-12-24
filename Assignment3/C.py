from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import nltk

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    tmp_s = { key : [ set(nltk.word_tokenize(el[1])[-window_size:] + nltk.word_tokenize(el[3])[:window_size]) for el in data[key] ] for key in data.keys() }
    s =  { key : [ item for sublist in tmp_s[key] for item in sublist ] for key in tmp_s.keys() }

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    for lexelt in data:
        set_i = nltk.word_tokenize(lexelt[1])[-10:] + nltk.word_tokenize(lexelt[3])[:10]
        vectors[lexelt[0]] = [ set_i.count(word) for word in set(s) ]

        labels[lexelt[0]] = lexelt[4]

    return vectors, labels

def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    selector = SelectKBest(chi2, k='all')
    selector.fit(X_train.values(), y_train.values())

    X_train_new = { key: value for key, value in zip(X_train.keys(), selector.transform(X_train.values())) }
    X_test_new = { key: value for key, value in zip(X_test.keys(), selector.transform(X_test.values())) }

    return X_train_new, X_test_new

# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    svm_clf.fit([ el for el in X_train.values() ], [ el for el in y_train.values() ])
    knn_clf.fit([ el for el in X_train.values() ], [ el for el in y_train.values() ])

    svm_results = [ (instance, label) for instance, label in zip(X_test, svm_clf.predict([ el for el in X_test.values() ])) ]
    knn_results = [ (instance, label) for instance, label in zip(X_test, knn_clf.predict([ el for el in X_test.values() ])) ]

    best_answer = svm_results

    return svm_results, knn_results, best_answer

# A.3, A.4 output
def print_results(results, output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    predictions = [ (key, el[0], el[1]) for key in results for el in results[key] ]
    predictions.sort(key=lambda tup: tup[1])
    output = [ replace_accented(el[0]) + " " + replace_accented(el[1]) + " " + el[2] + "\n" for el in predictions ]

    f = open(output_file,'w')
    for line in output:
        f.write(line)
    f.close()

# run part A
def run(train, test, language, knn_file, svm_file, best_answer_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    best_answer = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        # X_train_new, X_test_new = feature_selection(X_train, X_test, y_train)
        svm_results[lexelt], knn_results[lexelt], best_answer[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)
    print_results(best_answer, best_answer_file)
