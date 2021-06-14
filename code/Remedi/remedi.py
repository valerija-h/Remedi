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
    from keras.models import Model, Input, Sequential, model_from_json
    import keras as k
    from keras_contrib.layers import CRF
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, concatenate, InputSpec, TimeDistributed
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    from keras_contrib.utils import save_load_utils

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize

    # the GENIA and UMLS python wrappers
    from Libraries.geniatagger import GENIATagger
    tagger = GENIATagger(os.path.join(".", "Libraries/geniatagger-3.0.2", "geniatagger"))
    from pymetamap import MetaMap
    mm = MetaMap.get_instance('/Users/valerija/Downloads/public_mm/bin/metamap18') # set the file path of where you installed the public_mm folder

# to temp store information of concepts in a concept file - for easier pre-processing
class Concept:
    def __init__(self, word, lineNo, wordStart, wordEnd, con):
        self.word = word
        self.lineNo = lineNo
        self.wordStart = wordStart
        self.wordEnd = wordEnd
        self.con = con  # the tagged concept

def loadREModel(model_path, params, vocab_path):
    # load model from file
    json_file = open(model_path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path+".h5")
    # compile model with hyperparameters
    loaded_model.compile(loss=params['loss'], optimizer=params['optimization'], metrics=params['metrics'])
    # load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return loaded_model, vocab

def loadNERModel(model_path, params, vocab_path):
    # load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    model = buildNERModel(vocab, params)
    model.load_weights(model_path+".h5")
    return model, vocab

# tag each NP in a sentence using UMLS
def getUMLStags(words, NP):
    ST, CUI = ['O' for i in range(len(words))], ['O' for i in range(len(words))]
    for i, word in enumerate(words):
        if (NP[i] == 'B-NP'):
            current_i = i + 1
            current_word = word
            while (current_i < len(words)) and (NP[current_i] == 'I-NP'):
                current_word += ' '
                current_word += words[current_i]
                current_i += 1

            # get the tags of the word an mark it
            concepts, errors = mm.extract_concepts([current_word], [1])
            if (len(concepts) == 0): continue  # skip if no concepts found
            for j in range(i, current_i):
                ST[j] = concepts[0].semtypes
                CUI[j] = concepts[0].cui
    return ST, CUI

# get the POS and chunks and tags from the GENIA tagger
def getGENIAtags(sentence):
    chunks, POS, entities = [], [], []
    tags = tagger.tag(sentence)
    for word, base_form, pos_tag, chunk, named_entity in tags:
        chunks.append(chunk)
        POS.append(pos_tag)
        entities.append(named_entity)
    return chunks, POS, entities

def buildNERModel(vocab, params):
    n_words = len(vocab['words'])
    n_tags = len(vocab['concepts'])
    n_GENIA_POS = len(vocab['GENIA_POS'])
    n_GENIA_chunks = len(vocab['GENIA_chunks'])
    n_GENIA_entities = len(vocab['GENIA_entities'])
    n_UMLS_sem = len(vocab['UMLS_sem'])
    n_UMLS_cui = len(vocab['UMLS_cui'])
    word_embedding_size = params['word_embedding_size']
    dropout = params['dropout']
    recurrent_dropout = params['recurrent_dropout']
    activation = params['activation']
    global max_len

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

    return model

def getCommonFeatures(sentences):
    features = {
        'words': [],
        'GENIA_chunks': [],
        'GENIA_POS': [],
        'GENIA_entities': [],
        'UMLS_sem': [],
        'UMLS_cui': []
    }
    for sentence in sentences:
        sentence = sentence.strip('\n')
        words = nltk.word_tokenize(sentence)
        NLTK_POS = nltk.pos_tag(words)  # get POS tags - from NLTK
        GENIA_chunks, GENIA_POS, GENIA_entities = getGENIAtags(sentence)
        UMLS_sem, UMLS_cui = getUMLStags(words, GENIA_chunks)

        features['words'].append(words)
        features['GENIA_chunks'].append(GENIA_chunks)
        features['GENIA_POS'].append(GENIA_POS)
        features['GENIA_entities'].append(GENIA_entities)
        features['UMLS_sem'].append(UMLS_sem)
        features['UMLS_cui'].append(UMLS_cui)
    return features

def convertToIndexForm(features, vocab):
    indexForms = {}
    global max_len
    for key in features.keys():
        if key != 'e1_dist' and key != 'e2_dist':
            current_vocab = vocab[key]
            current_features = features[key]
            # get the index form and con
            idxform = {w: i for i, w in enumerate(current_vocab)}  
        if key == 'words':
            i2dxform = []
            for s in current_features:
                temp = []
                for w in s:
                    if w.lower() not in idxform.keys():
                        temp.append(idxform['UNK'])
                    else:
                        temp.append(idxform[w.lower()])
                i2dxform.append(temp)
        elif key == 'UMLS_sem' or key == 'UMLS_cui':
            i2dxform = []
            for s in current_features:
                temp = []
                for w in s:
                    if w not in idxform.keys():
                        temp.append(idxform['UNK'])
                    else:
                        temp.append(idxform[w])
                i2dxform.append(temp)
        elif key == 'e1_dist': 
            current_features = features[key]
            i2dxform = [s for s in current_features]
            indexForms[key] = pad_sequences(maxlen=max_len, sequences=i2dxform, padding="post",value=max_len-1)
            continue
        elif key == 'e2_dist': 
            current_features = features[key]
            i2dxform = [s for s in current_features]
            indexForms[key] = pad_sequences(maxlen=max_len, sequences=i2dxform, padding="post",value=max_len-1)
            continue
        else: i2dxform = [[idxform[w] for w in s] for s in current_features] 
        indexForms[key] = pad_sequences(maxlen=max_len, sequences=i2dxform, padding="post",value=len(current_vocab)-1)
    return indexForms

# convert entities in entity objects and return them for the current line number
def getEntities(concepts, words, lineNo):
    entities = []
    max_len = len(words)
    for i, word in enumerate(words):
        concept = concepts[i]
        if concept == "B-treatment" or concept == "B-problem":
            current_word = word
            current_pos = i
            wordStart = i
            while ((current_pos+1) < max_len) and (concepts[current_pos+1] == "I-treatment" or concepts[current_pos+1] == "I-problem"):
                current_pos += 1
                current_word = current_word + " " + words[current_pos]
            wordEnd = current_pos
            entities.append(Concept(current_word, lineNo, wordStart, wordEnd, concept[2:]))
    return entities

def getEntityPairs(entities):
    pairs = []
    for i, entity in enumerate(entities):
        if entity.con == "problem":
            for j, entity2 in enumerate(entities):
                if i == j: continue
                if entity2.con == "problem":
                    pairs.append([entity, entity2])
                if entity2.con == "treatment":
                    pairs.append([entity2, entity]) # put treatment first
    return pairs

def getDistance(entity_pair, words):
    e1, e2 = entity_pair[0], entity_pair[1]
    e1start, e2start, e1end, e2end = e1.wordStart, e2.wordStart, e1.wordEnd, e2.wordEnd
    e1dist, e2dist = [0 for i in range(len(words))], [0 for i in range(len(words))]

    for i in range(len(words)):
        if i >= e1start and i <= e1end:
            e1dist[i] = 0
        elif i < e1start:
            e1dist[i] = e1start - i
        else:
            e1dist[i] = i - e1end
        
        if i >= e2start and i <= e2end:
            e2dist[i] = 0
        elif i < e2start:
            e2dist[i] = e2start - i
        else:
            e2dist[i] = i - e2end
    
    return e1dist, e2dist

def getREFeatures(features, NER_entities):
    feature_words = features['words']
    new_features = {
        'words': [],
        'GENIA_chunks': [],
        'GENIA_POS': [],
        'GENIA_entities': [],
        'UMLS_sem': [],
        'UMLS_cui': [],
        'concepts': [],
        'e1_dist': [],
        'e2_dist': []
    }
    for i, words in enumerate(feature_words):
        concepts = NER_entities[i]
        entities = getEntities(concepts, words, i)
        entity_pairs = getEntityPairs(entities)

        if(len(entity_pairs) == 0): # skip sentences with no entity pairs
            print('No entity pairs found on sentence:' + str(i))
            continue

        for entity_pair in entity_pairs:
            # get the distance between words and entities
            e1_dist, e2_dist = getDistance(entity_pair, words)
            new_features['words'].append(words)
            new_features['GENIA_chunks'].append(features['GENIA_chunks'][i])
            new_features['GENIA_POS'].append(features['GENIA_POS'][i])
            new_features['GENIA_entities'].append(features['GENIA_entities'][i])
            new_features['UMLS_sem'].append(features['UMLS_sem'][i])
            new_features['UMLS_cui'].append(features['UMLS_cui'][i])
            new_features['concepts'].append(NER_entities[i])
            new_features['e1_dist'].append(e1_dist)
            new_features['e2_dist'].append(e2_dist)
    return new_features
                        

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

def getNEREntities(NER_model, features, vocab):
    model_input = [features['words'], features['GENIA_POS'], features['GENIA_chunks'], features['GENIA_entities'], features['UMLS_sem'], features['UMLS_cui']]
    prediction = NER_model.predict(model_input)

    # convert into an understandable format
    key = 'concepts'
    current_vocab = vocab[key]
    rels2idx = {w: i for i, w in enumerate(current_vocab)}
    idx2tag = {v: k for k, v in iteritems(rels2idx)}
    pred_labels = pred2label(prediction, idx2tag)
    
    return pred_labels

def getREEntities(RE_model, features, vocab):
    model_input = [features['words'], features['concepts'], features['GENIA_POS'], features['GENIA_chunks'], features['GENIA_entities'], features['UMLS_sem'], features['UMLS_cui'], features['e1_dist'], features['e2_dist']]
    prediction = RE_model.predict(model_input)

    # convert into an understandable format
    key = 'relations'
    current_vocab = vocab[key]
    rels2idx = {w: i for i, w in enumerate(current_vocab)}
    predictions = [np.argmax(pred) for pred in prediction]
    predictions = [list(rels2idx.keys())[list(rels2idx.values()).index(pred)] for pred in predictions]
    
    return predictions

def parseEntities(commonFeatures, NER_entities, REFeatures, RE_entities):
    for i, sentence in enumerate(commonFeatures['words']):
        # construct sentence
        con_sentence = ' '.join(word for word in sentence)
        print('The following entities were found in sentence: \"' + con_sentence + '\"')
        for j, concept in enumerate(NER_entities[i]):
            if concept == 'B-problem' or concept == 'B-treatment':
                concept_type = concept[2:]
                current_word = sentence[j]
                current_pos = j
                while current_pos + 1 < len(sentence) and NER_entities[i][current_pos + 1] != 'O':
                    current_pos += 1
                    current_word += (' ' + sentence[current_pos])
                print('Concept Type: \"' + concept_type + '\"\t\tConcept Word: \"' + current_word + '\"')

    print('\nThe following relations were found in the document:')       
    for i, sentence in enumerate(REFeatures['words']):
        relation = RE_entities[i]
        if relation != 'NPPR' and relation != 'NTPR':
            con_sentence = ' '.join(word for word in sentence)
            e1_word, e2_word, e1_type, e2_type = '', '', '', ''

            for j, dist in enumerate(REFeatures['e1_dist'][i]):
                if dist == 0:
                    e1_type = REFeatures['concepts'][i][j][2:]
                    e1_word = sentence[j]
                    current_pos = j
                    while current_pos + 1 < len(sentence) and REFeatures['e1_dist'][i][current_pos + 1] == 0:
                        current_pos += 1
                        e1_word += (' ' + sentence[current_pos])
                    break
            
            for j, dist in enumerate(REFeatures['e2_dist'][i]):
                if dist == 0:
                    e2_type = REFeatures['concepts'][i][j][2:]
                    e2_word = sentence[j]
                    current_pos = j
                    while current_pos + 1 < len(sentence) and REFeatures['e2_dist'][i][current_pos + 1] == 0:
                        current_pos += 1
                        e2_word += (' ' + sentence[current_pos])
                    break

        
            print('Relation: \"' + relation + '\"\t\tE1 Type: \"' + e1_type + '\", E1 Word: \"' + e1_word + '\"\tE2 Type: \"' + e2_type + '\", E2 Word: \"' + e2_word + '\"')

# parameters to change
NER_model_path = 'Models/NER-model'
NER_vocabulary_path = 'Models/NER-vocab.pkl'
RE_model_path = 'Models/RE-model'
RE_vocabulary_path = 'Models/RE-vocab.pkl'
input_path = 'input.txt'
max_len = 201

NER_params = {
    'optimization': 'rmsprop',
    'word_embedding_size': 250,
    'dropout': 0.1,
    'recurrent_dropout': 0.2,
    'activation': 'elu'
}

RE_params = {
    'optimization': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}                          

print('Loading Models and Vocabularies....')
NER_model, NER_vocab = loadNERModel(NER_model_path, NER_params, NER_vocabulary_path)
RE_model, RE_vocab = loadREModel(RE_model_path, RE_params, RE_vocabulary_path)
print('Finished Loading Models and Vocabularies.')

with open(input_path, 'r') as f:
    sentences = f.readlines()
    f.close()
commonFeatures = getCommonFeatures(sentences)
commonNERFeatures = convertToIndexForm(commonFeatures, NER_vocab)
NER_entities = getNEREntities(NER_model, commonNERFeatures, NER_vocab)

REFeatures = getREFeatures(commonFeatures, NER_entities)
if len(REFeatures['words']) > 0:
    commonREFeatures = convertToIndexForm(REFeatures, RE_vocab)
    RE_entities = getREEntities(RE_model, commonREFeatures, RE_vocab)
    parseEntities(commonFeatures, NER_entities, REFeatures, RE_entities)
else:
    print('No entity pairs found hence no relations can be found.')

