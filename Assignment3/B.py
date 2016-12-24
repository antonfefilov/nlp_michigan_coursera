import A
import nltk
import string
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

# You might change the window size
window_size = 5

def preprocess_context(context, language):
    # remove punctuations
    table = { ord(c): None for c in string.punctuation }
    context = context.translate(table)

    # tokenize and stem
    porter = PorterStemmer()
    stems = [ porter.stem(word) for word in nltk.word_tokenize(context) ]

    # remove stop words
    if language == "English":
        stop = stopwords.words('english')
    else:
        stop = stopwords.words('spanish')
    tokens = [ token for token in stems if token not in stop ]

    return tokens

def build_s(data, language):
    s = []

    for el in data:
        left_tokens = preprocess_context(el[1], language)
        right_tokens = preprocess_context(el[3], language)

        s += list( set(left_tokens[-window_size:] + right_tokens[:window_size]) )

    return s

def simple_window(s, left_tokens, right_tokens):
    features = {}

    set_i = left_tokens[-window_size:] + right_tokens[:window_size]
    features = { "sw_" + str(index): set_i.count(word) for word, index in zip(s, range(len(s))) }

    return features

def collocational(left_tokens, head, right_tokens):
    features = {}

    word_0 = head

    left_tokens = left_tokens[-2:]
    if len(left_tokens) < 1:
        word_2_l, word_1_l = ["NaN", "NaN"]
        pos_2_l = "NaN"
        pos_1_l = "NaN"
    elif len(left_tokens) < 2:
        word_2_l, word_1_l = ["NaN", left_tokens[0]]
        pos_2_l = "NaN"
        pos_1_l = nltk.pos_tag([word_1_l])[0][1]
    else:
        word_2_l, word_1_l = left_tokens
        pos_2_l = nltk.pos_tag([word_2_l])[0][1]
        pos_1_l = nltk.pos_tag([word_1_l])[0][1]

    right_tokens = right_tokens[:2]
    if len(right_tokens) < 1:
        word_1_r, word_2_r = ["NaN", "NaN"]
        pos_1_r = "NaN"
        pos_2_r = "NaN"
    elif len(right_tokens) < 2:
        word_1_r, word_2_r = [right_tokens[0], "NaN"]
        pos_1_r = nltk.pos_tag([word_1_r])[0][1]
        pos_2_r = "NaN"
    else:
        word_1_r, word_2_r = right_tokens
        pos_1_r = nltk.pos_tag([word_1_r])[0][1]
        pos_2_r = nltk.pos_tag([word_2_r])[0][1]

    pos_0 = nltk.pos_tag([word_0])[0][1]

    features = {
            "word-2": word_2_l,
            "word-1": word_1_l,
            "word_0 word+1": word_0 + " " + word_1_r,
            "word+2": word_2_r,
            "pos-2": pos_2_l,
            "pos-1": pos_1_l,
            "pos_0 pos+1": pos_0 + " " + pos_1_r,
            "pos+2": pos_2_r
            }

    return features

def synsets(word):
    synsets = wn.synsets(word)

    features = { "syn_" + str(synsets.index(s)): s.name() for s in synsets  }

    return features

# B.1.a,b,c,d
def extract_features(data, language):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # for simple_window
    s = build_s(data, language)

    for lexelt in data:
        labels[lexelt[0]] = lexelt[4]

        left_tokens = preprocess_context(lexelt[1], language)
        right_tokens = preprocess_context(lexelt[3], language)
        head = lexelt[2]

        features[lexelt[0]] = {}
        features[lexelt[0]].update( simple_window(s, left_tokens, right_tokens) )
        features[lexelt[0]].update( collocational(left_tokens, head, right_tokens) )
        features[lexelt[0]].update( synsets(head) )

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
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

    selector = SelectKBest(chi2)
    selector.fit(X_train.values(), y_train.values())

    X_train_new = { key: value for key, value in zip(X_train.keys(), selector.transform(X_train.values())) }
    X_test_new = { key: value for key, value in zip(X_test.keys(), selector.transform(X_test.values())) }

    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    # knn_clf = neighbors.KNeighborsClassifier()
    # forest_clf = RandomForestClassifier(n_estimators=100, max_features=None, n_jobs=-1, min_samples_split=1, random_state=50)
    svm_clf = svm.LinearSVC()

    # knn_clf.fit([ el for el in X_train.values() ], [ el for el in y_train.values() ])
    # forest_clf.fit([ el for el in X_train.values() ], [ el for el in y_train.values() ])
    svm_clf.fit([ el for el in X_train.values() ], [ el for el in y_train.values() ])

    results = [ (instance, label) for instance, label in zip(X_test, svm_clf.predict([ el for el in X_test.values() ])) ]

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:
        train_features, y_train = extract_features(train[lexelt], language)
        test_features, _ = extract_features(test[lexelt], language)

        X_train, X_test = vectorize(train_features, test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test, y_train)
        results[lexelt] = classify(X_train_new, X_test_new, y_train)

    A.print_results(results, answer)
