import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import os
    import re  # for regular expression extraction
    import nltk  # for tokenization
    import pickle
    import sys
    import numpy as np
    import random
    from future.utils import iteritems

    # for building an evaluating a CRF model
    import matplotlib.pyplot as plt
    from keras.callbacks import ModelCheckpoint
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from keras.models import Model, Input
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, InputSpec
    import keras as k
    from keras_contrib.layers import CRF
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from seqeval.metrics import accuracy_score
    from seqeval.metrics import classification_report
    from seqeval.metrics import f1_score

    from talos import Scan
    from talos import Evaluate
    from talos import Deploy
    from talos import Restore


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

# build a list of unique words and concepts
def getVocabularies(sentences):
    vocabularies = {
        'words': [],
        'concepts': [],
        'NLTK_POS': [] ,
        'GENIA_POS': [],
        'GENIA_chunks': [],
        'GENIA_entities': [],
        'UMLS_sem': [],
        'UMLS_cui': []
    }
    max_len = 0 # max number of words in a sentence
    for sent in sentences:
        if len(sent.words) > max_len: max_len = len(sent.words)
        for i, word in enumerate(sent.words):
            word = word.lower() # use lower case words
            concept = sent.concepts[i]
            NLTK_POS = sent.NLTK_POS[i]
            GENIA_POS = sent.GENIA_POS[i]
            GENIA_chunks = sent.GENIA_chunks[i]
            GENIA_entities = sent.GENIA_entities[i]
            UMLS_sem = sent.UMLS_sem[i]
            UMLS_cui = sent.UMLS_cui[i]

            if word not in vocabularies['words']: vocabularies['words'].append(word)
            if concept not in vocabularies['concepts'] : vocabularies['concepts'].append(concept)
            if NLTK_POS not in vocabularies['NLTK_POS']: vocabularies['NLTK_POS'].append(NLTK_POS)
            if GENIA_POS not in vocabularies['GENIA_POS']: vocabularies['GENIA_POS'].append(GENIA_POS)
            if GENIA_chunks not in vocabularies['GENIA_chunks']: vocabularies['GENIA_chunks'].append(GENIA_chunks)
            if GENIA_entities not in vocabularies['GENIA_entities']: vocabularies['GENIA_entities'].append(GENIA_entities)
            if UMLS_sem not in vocabularies['UMLS_sem']: vocabularies['UMLS_sem'].append(UMLS_sem)
            if UMLS_cui not in vocabularies['UMLS_cui']: vocabularies['UMLS_cui'].append(UMLS_cui)
    return vocabularies, max_len

# get the i2dx form of all the vocabularies
def getIndexForm(vocabularies):
    i2dxforms = {
        'words':            {w: i for i, w in enumerate(vocabularies['words'])},
        'concepts':         {c: i for i, c in enumerate(vocabularies['concepts'])},
        'NLTK_POS':         {p: i for i, p in enumerate(vocabularies['NLTK_POS'])},
        'GENIA_POS':        {p: i for i, p in enumerate(vocabularies['GENIA_POS'])},
        'GENIA_chunks':     {c: i for i, c in enumerate(vocabularies['GENIA_chunks'])},
        'GENIA_entities':   {e: i for i, e in enumerate(vocabularies['GENIA_entities'])},
        'UMLS_sem':         {s: i for i, s in enumerate(vocabularies['UMLS_sem'])},
        'UMLS_cui':         {c: i for i, c in enumerate(vocabularies['UMLS_cui'])}
    }
    return i2dxforms

# convert all vocabularies to their idx form
def transformToIndexForm(sentences, i2dxforms):
    indexedForms = {
        'words':            [[i2dxforms['words'][w.lower()] for w in s.words] for s in sentences],
        'concepts':         [[i2dxforms['concepts'][c] for c in s.concepts] for s in sentences],
        'NLTK_POS':         [[i2dxforms['NLTK_POS'][p] for p in s.NLTK_POS] for s in sentences],
        'GENIA_POS':        [[i2dxforms['GENIA_POS'][p] for p in s.GENIA_POS] for s in sentences],
        'GENIA_chunks':     [[i2dxforms['GENIA_chunks'][c] for c in s.GENIA_chunks] for s in sentences],
        'GENIA_entities':   [[i2dxforms['GENIA_entities'][e] for e in s.GENIA_entities] for s in sentences],
        'UMLS_sem':         [[i2dxforms['UMLS_sem'][s] for s in s.UMLS_sem] for s in sentences],
        'UMLS_cui':         [[i2dxforms['UMLS_cui'][c] for c in s.UMLS_cui] for s in sentences]
    }
    return indexedForms

# converts labels into a format for outputting results
def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

# display the f1-score and classification report for the results of a model
def evaluateModel(i2dxforms, test_pred, y_test):
    idx2tag = {v: k for k, v in iteritems(i2dxforms['concepts'])}

    pred_labels = pred2label(test_pred, idx2tag)
    test_labels = pred2label(y_test, idx2tag)

    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print(classification_report(test_labels, pred_labels, digits=3)) 


def trainMultipleModel(params, data):
    word_embedding_size = params['word_embedding_size']
    recurrent_dropout = params['recurrent_dropout']
    dropout = params['dropout']
    batch_size = params['batch_size']
    epochs = 5

    X_train = data['X_train']
    y_train = data['y_train']
    X_GENIA_POS_train = data['X_GENIA_POS_train']
    X_GENIA_chunks_train = data['X_GENIA_chunks_train']
    X_GENIA_entities_train = data['X_GENIA_entities_train']
    X_UMLS_sem_train = data['X_UMLS_sem_train']
    X_UMLS_cui_train = data['X_UMLS_cui_train']
    n_words = data['n_words']
    n_tags = data['n_tags']
    n_GENIA_POS = data['n_GENIA_POS']
    n_GENIA_chunks = data['n_GENIA_chunks']
    n_GENIA_entities = data['n_GENIA_entities']
    n_UMLS_sem = data['n_UMLS_sem']
    n_UMLS_cui = data['n_UMLS_cui']

    # --------------------- Defining the Model -----------------------
    input_seq =             Input(shape=(max_len,))
    embed_seq =             Embedding(input_dim=n_words+1, output_dim=word_embedding_size, input_length=max_len)(input_seq)

    input_seq_GENIA_POS =   Input(shape=(max_len,))
    embed_seq_GENIA_POS =   Embedding(input_dim=n_GENIA_POS+1, output_dim=word_embedding_size, input_length=max_len)(input_seq_GENIA_POS)

    input_seq_GENIA_chunks = Input(shape=(max_len,))
    embed_seq_GENIA_chunks = Embedding(input_dim=n_GENIA_chunks+1, output_dim=word_embedding_size, input_length=max_len)(input_seq_GENIA_chunks)

    input_seq_GENIA_entities = Input(shape=(max_len,))
    embed_seq_GENIA_entities = Embedding(input_dim=n_GENIA_entities+1, output_dim=word_embedding_size, input_length=max_len)(input_seq_GENIA_entities)

    input_seq_UMLS_sem =    Input(shape=(max_len,))
    embed_seq_UMLS_sem =    Embedding(input_dim=n_UMLS_sem+1, output_dim=word_embedding_size, input_length=max_len)(input_seq_UMLS_sem)

    input_seq_UMLS_cui =    Input(shape=(max_len,))
    embed_seq_UMLS_cui =    Embedding(input_dim=n_UMLS_cui+1, output_dim=word_embedding_size, input_length=max_len)(input_seq_UMLS_cui)

    model = concatenate([embed_seq, embed_seq_GENIA_POS, embed_seq_GENIA_chunks, embed_seq_GENIA_entities, embed_seq_UMLS_sem, embed_seq_UMLS_cui])
    model = Bidirectional(LSTM(units=word_embedding_size, 
                            return_sequences=True,
                            dropout= dropout, 
                            recurrent_dropout=recurrent_dropout))(model)
    # TimeDistributed Layer
    model = TimeDistributed(Dense(n_tags, activation=params['activation']))(model)
    # CRF layer
    crf = CRF(n_tags)
    out = crf(model)  # output
    model = Model([input_seq, input_seq_GENIA_POS, input_seq_GENIA_chunks, input_seq_GENIA_entities, input_seq_UMLS_sem, input_seq_UMLS_cui], out) 

    model.compile(optimizer=params['optimization'], loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
    # model.summary()

    # --------------------- Training the Model -----------------------
    history = model.fit([X_train, X_GENIA_POS_train, X_GENIA_chunks_train, X_GENIA_entities_train, X_UMLS_sem_train, X_UMLS_cui_train], np.array(y_train), batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)

    return history, model


def buildMultipleFeatureModel(vocabularies, indexedForms, i2dxforms, max_len):
    n_words = len(vocabularies['words'])
    n_tags = len(vocabularies['concepts'])
    n_GENIA_POS = len(vocabularies['GENIA_POS'])
    n_GENIA_chunks = len(vocabularies['GENIA_chunks'])
    n_GENIA_entities = len(vocabularies['GENIA_entities'])
    n_UMLS_sem = len(vocabularies['UMLS_sem'])
    n_UMLS_cui = len(vocabularies['UMLS_cui'])

    X = indexedForms['words']
    X_GENIA_POS = indexedForms['GENIA_POS']
    X_GENIA_chunks = indexedForms['GENIA_chunks']
    X_GENIA_entities = indexedForms['GENIA_entities']
    X_UMLS_sem = indexedForms['UMLS_sem']
    X_UMLS_cui = indexedForms['UMLS_cui']
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post",value=n_words-1)
    X_GENIA_POS = pad_sequences(maxlen=max_len, sequences=X_GENIA_POS, padding="post",value=n_GENIA_POS-1)
    X_GENIA_chunks = pad_sequences(maxlen=max_len, sequences=X_GENIA_chunks, padding="post",value=n_GENIA_chunks-1)
    X_GENIA_entities = pad_sequences(maxlen=max_len, sequences=X_GENIA_entities, padding="post",value=n_GENIA_entities-1)
    X_UMLS_sem = pad_sequences(maxlen=max_len, sequences=X_UMLS_sem, padding="post",value=n_UMLS_sem-1)
    X_UMLS_cui = pad_sequences(maxlen=max_len, sequences=X_UMLS_cui, padding="post",value=n_UMLS_cui-1)

    y = indexedForms['concepts']
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=i2dxforms['concepts']["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    # split train and test data
    X_train, X_test, X_GENIA_POS_train, X_GENIA_POS_test, X_GENIA_chunks_train, X_GENIA_chunks_test, X_GENIA_entities_train, X_GENIA_entities_test,  X_UMLS_sem_train, X_UMLS_sem_test, X_UMLS_cui_train, X_UMLS_cui_test, y_train, y_test = train_test_split(X, X_GENIA_POS, X_GENIA_chunks, X_GENIA_entities, X_UMLS_sem, X_UMLS_cui, y, test_size=0.2)
    
    training_labels = [X_test, X_GENIA_POS_test, X_GENIA_chunks_test, X_GENIA_entities_test, X_UMLS_sem_test, X_UMLS_cui_test]

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_GENIA_POS_train': X_GENIA_POS_train,
        'X_GENIA_chunks_train': X_GENIA_chunks_train,
        'X_GENIA_entities_train': X_GENIA_entities_train,
        'X_UMLS_sem_train': X_UMLS_sem_train,
        'X_UMLS_cui_train': X_UMLS_cui_train,
        'n_words': n_words,
        'n_tags': n_tags,
        'n_GENIA_POS': n_GENIA_POS,
        'n_GENIA_chunks': n_GENIA_chunks,
        'n_GENIA_entities': n_GENIA_entities,
        'n_UMLS_sem': n_UMLS_sem,
        'n_UMLS_cui': n_UMLS_cui
    }
    
    counter = 0
    for word_embedding_size in [50, 100, 150, 200, 250, 300]:
        params={
            'word_embedding_size': word_embedding_size,
            'batch_size': 32,
            'recurrent_dropout': 0.2,
            'dropout': 0.1,
            'optimization': 'rmsprop',
            'activation': 'elu'
        }
        history, model = trainMultipleModel(params, data)
        counter += 1
        print("Current params:", params)
        print("WME:", word_embedding_size)
        print("Counter:", counter, "/", 6)
        # --------------------- Evaluating the Model -----------------------
        test_pred = model.predict(training_labels, verbose=1)
        evaluateModel(i2dxforms, test_pred, y_test)


# paths of training data pre-processed
paths = [ '../Preprocessed Dataset/beth-data-NT.pkl', '../Preprocessed Dataset/partners-data-NT.pkl', '../Preprocessed Dataset/test-data-NT.pkl']
noinfo_no = -1  # number of non-informative sentences

documents = getDocuments(paths)
sentences = getInformativeSet(documents, noinfo_no)
vocabularies, max_len = getVocabularies(sentences)  # get a list of individual words
i2dxforms = getIndexForm(vocabularies)
indexedForms = transformToIndexForm(sentences, i2dxforms)

buildMultipleFeatureModel(vocabularies, indexedForms, i2dxforms, max_len)