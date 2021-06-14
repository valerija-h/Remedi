import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import os
    import re  # for regular expression extraction
    import nltk  # for tokenization
    import pickle
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    import random
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    # for building an evaluating a CRF model
    import sklearn
    import scipy.stats
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
    import sklearn_crfsuite
    from sklearn_crfsuite import scorers
    from sklearn_crfsuite import metrics

class Sentence:
    def __init__(self, words, concepts, NLTK_POS, GENIA_POS, GENIA_chunks, GENIA_entities, UMLS_sem, UMLS_cui):
        self.words = words
        self.concepts = concepts
        self.NLTK_POS = NLTK_POS
        self.GENIA_POS = GENIA_POS
        self.GENIA_chunks = GENIA_chunks
        self.GENIA_entities = GENIA_entities
        self.UMLS_sem = UMLS_sem
        self.UMLS_cui = UMLS_cui

def getDocuments(paths):
    documents = []
    for path in paths:
        with open(path, 'rb') as f:
            batch = pickle.load(f)
            documents += batch
    return documents

# get all sentences and features from documents then split them into informative and noninformative sets
def getInformativeSet(documents, noninfo_no):
    info, non_info = [], []
    for document in documents:
        for i in range(len(document['sentences'])):
            sentence = Sentence(document['words'][i], document['concepts'][i], document['NLTK_POS'][i],
                                document['GENIA_POS'][i], document['GENIA_chunks'][i], document['GENIA_entities'][i],
                                document['UMLS_sem'][i], document['UMLS_cui'][i]
                                )
            if document['informative'][i] == True:
                info.append(sentence)
            else:
                non_info.append(sentence)
    # shuffle the non informative set and take the first n candidates
    if non_info != -1:
        random.shuffle(non_info)
        non_info = non_info[:noninfo_no]
    sentences = info + non_info
    random.shuffle(sentences)  # combine informative and non-informative sentences then shuffle
    return sentences

# given a sentence and an index - get the features for a given word
def word2features(sent, i):
    word = sent.words[i]
    NLTK_POS = sent.NLTK_POS[i]
    GENIA_POS = sent.GENIA_POS[i]
    GENIA_chunks = sent.GENIA_chunks[i]
    GENIA_entities = sent.GENIA_entities[i]
    UMLS_sem = sent.UMLS_sem[i]
    UMLS_cui = sent.UMLS_cui[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'nltk.postag': NLTK_POS,
        'nltk.postag[:2]': NLTK_POS[:2],
        'genia.postag': GENIA_POS,
        'genia.chunks': GENIA_chunks,
        'genia.entities': GENIA_entities,
        'UMLS.sem': UMLS_sem,
        'UMLS.cui': UMLS_cui
    }

    if i > 0:  # grab features of word before it
        word1 = sent.words[i-1]
        NLTK_POS1 = sent.NLTK_POS[i-1]
        GENIA_POS1 = sent.GENIA_POS[i-1]
        GENIA_chunks1 = sent.GENIA_chunks[i-1]
        GENIA_entities1 = sent.GENIA_entities[i-1]
        UMLS_sem1 = sent.UMLS_sem[i-1]
        UMLS_cui1 = sent.UMLS_cui[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:nltk.postag': NLTK_POS1,
            '-1:nltk.postag[:2]': NLTK_POS1[:2],
            '-1:genia.postag': GENIA_POS1,
            '-1:genia.chunks': GENIA_chunks1,
            '-1:genia.entities': GENIA_entities1,
            '-1:UMLS.sem': UMLS_sem1,
            '-1:UMLS.cui': UMLS_cui1
        })
    else:
        features['BOS'] = True  # beginning of sentence

    if i < len(sent.words)-1:
        word1 = sent.words[i+1]
        NLTK_POS1 = sent.NLTK_POS[i+1]
        GENIA_POS1 = sent.GENIA_POS[i+1]
        GENIA_chunks1 = sent.GENIA_chunks[i+1]
        GENIA_entities1 = sent.GENIA_entities[i+1]
        UMLS_sem1 = sent.UMLS_sem[i+1]
        UMLS_cui1 = sent.UMLS_cui[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:nltk.postag': NLTK_POS1,
            '+1:nltk.postag[:2]': NLTK_POS1[:2],
            '+1:genia.postag': GENIA_POS1,
            '+1:genia.chunks': GENIA_chunks1,
            '+1:genia.entities': GENIA_entities1,
            '+1:UMLS.sem': UMLS_sem1,
            '+1:UMLS.cui': UMLS_cui1
        })
    else:
        features['EOS'] = True

    return features

# return a list of features for a given sentence
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent.words))]

# return a list of labels/concepts for a given sentence
def sent2labels(sent):
    return [label for label in sent.concepts]

#  --------- bag of words attempt - to do in future versions
# def getBOWvocabulary(sentences):
#     vocab = []
#     # for each sentence, extract words, remove stopwords
#     for sentence in sentences:
#         cleaned_words = [w.lower() for w in sentence.words if w not in stop_words]
#         vocab.extend(cleaned_words)
#     return sorted(list(set(vocab)))
   
# def getBOWfeature(sentences):
#     vocab = getBOWvocabulary(sentences)

#     for sentence in sentences:
#         words = [w.lower() for w in sentence.words if w not in stop_words]
#         bag_vector = numpy.zeros(len(vocab))
#         for w in words:
#             for i, word in enumerate(vocab):
#                 if(word == w):
#                     bag_vector[i] += 1
        
#         print("{0} \n{1}\n".format(sentence.words,numpy.array(bag_vector)))
# getBOWfeature(sentences)

def buildSingleModel(X_train, X_test, y_train, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='pa',
        c=2,
        pa_type=2,
        max_iterations=100
    )

    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    # save the model to disk
    filename = '../Models/New-CRF-Model.sav'
    pickle.dump(crf, open(filename, 'wb'))

# Use GridSearchCV to determine the best CRF model
def findBestHyperParameters(X_train, X_test, y_train, y_test):
    # labels = ['B-treatment', 'I-treatment', 'B-problem', 'I-problem', 'B-test', 'I-test']  # NER labels
    labels = ['B-treatment', 'I-treatment', 'B-problem', 'I-problem']  # NER labels

    crf = sklearn_crfsuite.CRF(
        max_iterations=100,
        all_possible_transitions=True
    )

    hyper_params = [
        {
            'algorithm': ['lbfgs'],
            'c1': [0.001, 0.01, 0.1, 0],
            'c2': [0.001, 0.01, 0.1, 0],
            'epsilon': [0.000001, 0.00001, 0.0001],
            'delta': [0.000001, 0.00001, 0.0001],
            'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking']
        },
        {
            'algorithm': ['l2sgd'],
            'c2': [0.001, 0.01, 0.1, 0],
            'calibration_eta': [0.01, 0.1, 1],
            'calibration_rate': [1, 2, 3]
        },
        {
            'algorithm': ['pa'],
            'pa_type': [0, 1, 2],
            'c': [0.5, 1, 2]
        },
        {
            'algorithm': ['arow'],
            'gamma': [0.01, 0.1, 1]
        }
    ]

    # use f1 score for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

    model_cv = GridSearchCV(crf, param_grid=hyper_params, scoring=f1_scorer, cv=3, verbose=10, n_jobs=-1)
    print("Training Model.....")
    model_cv.fit(X_train, y_train)
    
    pkl_filename = "../Models/NER-CRF-GridSearchCV-NT.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model_cv, file)

    print('best params:', model_cv.best_params_)
    print('best CV score:', model_cv.best_score_)
    print('model size: {:0.2f}M'.format(model_cv.best_estimator_.size_ / 1000000))

# Use RandomSearchCV for a faster variation and explore more continous variables
def buildMultipleModels(X_train, X_test, y_train, y_test):
    # labels = ['B-treatment', 'I-treatment', 'B-problem', 'I-problem', 'B-test', 'I-test']  # NER labels
    labels = ['B-treatment', 'I-treatment', 'B-problem', 'I-problem']  # NER labels
    
    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        max_iterations=100,
        all_possible_transitions=True
    )

    params_space = {
        'algorithm': ['lbfgs'],
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        'epsilon': [0.000001, 0.00001, 0.0001],
        'delta': [0.000001, 0.00001, 0.0001],
        'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking']
    }
    
    # use f1 score for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    
    # search
    model = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=5,
                            n_jobs=-1,
                            n_iter=2,
                            scoring=f1_scorer)
    model.fit(X_train, y_train)
    
    pkl_filename = "../Models/NER-CRF-RandomizedSearchCV.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    # crf = rs.best_estimator_
    print('best params:', model.best_params_)
    print('best CV score:', model.best_score_)
    print('model size: {:0.2f}M'.format(model.best_estimator_.size_ / 1000000))


# paths of training data pre-processed
paths = [ '../Preprocessed Dataset/beth-data-NT.pkl', '../Preprocessed Dataset/partners-data-NT.pkl', '../Preprocessed Dataset/test-data-NT.pkl']
noinfo_no = -1  # number of non-informative sentences to keep

documents = getDocuments(paths)
sentences = getInformativeSet(documents, noinfo_no)

# split sentences into a training and testing set them convert into features
train_sents, test_sents = train_test_split(sentences, test_size=0.2, random_state=42, shuffle=True)
# convert them into features and labels for the model
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

buildSingleModel(X_train, X_test, y_train, y_test) # build a single model and store it in Models
# buildMultipleModels(X_train, X_test, y_train, y_test) # hyperparameter optimization with RandomizedSearchCV
# findBestHyperParameters(X_train, X_test, y_train, y_test) # hyperparameter optimization with GridSearchCV