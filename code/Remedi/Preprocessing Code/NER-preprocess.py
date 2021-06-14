import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

import os
import re  # for regular expression extraction
import nltk  # for tokenization
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# the GENIA and UMLS python wrappers
from Libraries.geniatagger import GENIATagger
tagger = GENIATagger(os.path.join(".", "Libraries/geniatagger-3.0.2", "geniatagger"))
from pymetamap import MetaMap
mm = MetaMap.get_instance('/Users/valerija/Downloads/public_mm/bin/metamap18') # set the file path of where you installed the public_mm folder

# paths of the concept data and text and relations - training data
paths = [
    ('../Dataset/beth/txt/', '../Dataset/beth/concept/', '../Dataset/beth/rel/'),
    ('../Dataset/partners/txt/', '../Dataset/partners/concept/', '../Dataset/partners/rel/'),
    ('../Dataset/test/txt/', '../Dataset/test/concepts/', '../Dataset/test/rel/')
]

# to temp store information of concepts in a concept file - for easier pre-processing
class Concept:
    def __init__(self, word, lineNo, wordStart, wordEnd, con):
        self.word = word
        self.lineNo = lineNo
        self.wordStart = wordStart
        self.wordEnd = wordEnd
        self.con = con  # the tagged concept

class Document:
    def __init__(self, doc_name):
        self.doc_name = doc_name
        self.sentences = []
        self.words = []  # parsed sentences
        self.concepts = []  # NER tokens labelled
        self.NLTK_POS = []
        self.GENIA_POS = []
        self.GENIA_chunks = []
        self.GENIA_entities = []
        self.UMLS_sem = []  # semantic types
        self.UMLS_cui = []  # concept identifiers
        self.informative = []  # is the sentence informative or non-informative (has no tagged NER)

    # adding one sentence and its features to the document
    def update(self, sentence, words, concepts, NLTK_POS, GENIA_POS, GENIA_chunks, GENIA_entities, UMLS_sem, UMLS_cui, informative):
        self.sentences.append(sentence)
        self.words.append(words)
        self.concepts.append(concepts)
        self.NLTK_POS.append(NLTK_POS)
        self.GENIA_POS.append(GENIA_POS)
        self.GENIA_chunks.append(GENIA_chunks)
        self.GENIA_entities.append(GENIA_entities)
        self.UMLS_sem.append(UMLS_sem)
        self.UMLS_cui.append(UMLS_cui)
        self.informative.append(informative)

# give it the concept sentences of a file and returns a list of Concept objects
def parseConcepts(conSentences):
    final = []
    for sentence in conSentences:
        # get the word, line number, word start and word end and tag
        frags = sentence.strip('\n').split("||")
        word = re.search('"(.*)"', frags[0]).group(1)  # get text between first and last \"
        con = re.search('"(.*)"', frags[1]).group(1)

        ####### Assumptions skip test relations
        if con == "test":
            continue

        wordplacements = frags[0].replace('c=\"' + word + '\"', '').strip(' ').split(
            ' ')  # remove the concept from frags [0] and split by ' ' to get "4:6 4:7" format
        lineno = int(wordplacements[0].split(':')[0])
        wordstart = int(wordplacements[0].split(':')[1])
        wordend = int(wordplacements[1].split(':')[1])
        final.append(Concept(word, lineno - 1, wordstart, wordend, con))
    return final

# only get entities or concepts for the current line number
def getEntities(lineNo, concepts):
    final = []
    for concept in concepts:
        if lineNo == concept.lineNo:
            final.append(concept)
    return final

# tag each word in a given sentence in the BIO format
def tagConcepts(words, entities):
    final = ['O' for word in range(len(words))]
    for entity in entities:
        final[entity.wordStart] = 'B-' + entity.con
        currentwordno = entity.wordStart
        currentwordno += 1
        while (currentwordno != entity.wordEnd + 1):
            final[currentwordno] = 'I-' + entity.con
            currentwordno += 1
    return final

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

# input = textPath, conceptPath
def getSentences(textP, conP):
    counter = 0 # store how many documents have been go throughs
    documents = []  # to store a list of Document objects
    for r, d, f in os.walk(textP):
        for txtfile in f:
            if txtfile.endswith(".txt"):
                filename = os.path.splitext(txtfile)[0]  # get name of file without extension
                document = Document(filename)  # create a new document object

                textPath = textP + filename + '.txt'
                conPath = conP + filename + '.con'

                # open files and get all sentences
                with open(textPath) as f:
                    textSentences = f.readlines()
                with open(conPath) as f:
                    conSentences = f.readlines()

                # parse concept file into list of objects
                concepts = parseConcepts(conSentences)

                # for each sentence in the text file
                for lineNo, textSentence in enumerate(textSentences):
                    # get all concepts objects for the current line number
                    entities = getEntities(lineNo, concepts)

                    # get the words and labels and POS
                    sentence = textSentence.strip('\n')
                    words = sentence.split(' ')
                    words = list(filter(None, words))  # remove empty strings

                    # getting POS and GENIA features and UMLS features
                    NLTK_POS = nltk.pos_tag(words)  # get POS tags - from NLTK
                    NLTK_POS = [pos[1] for pos in NLTK_POS]
                    GENIA_chunks, GENIA_POS, GENIA_entities = getGENIAtags(sentence)
                    UMLS_sem, UMLS_cui = getUMLStags(words, GENIA_chunks)  # tag noun phrases with UMLS

                    cons = tagConcepts(words, entities)  # tag words in BIO format

                    if len(entities) > 0:
                        informative = True
                    else:
                        informative = False
                    document.update(sentence, words, cons, NLTK_POS, GENIA_POS, GENIA_chunks, GENIA_entities, UMLS_sem, UMLS_cui, informative)

                documents.append(document)
                counter += 1
                print("\n\n############### Document completed:", filename)
                print("No. of documents completed:", counter, "/", len(os.listdir(textP)))
                
    return documents

# variables to change
save_paths = [ '../Preprocessed Dataset/beth-data-NT.pkl', '../Preprocessed Dataset/partners-data-NT.pkl', '../Preprocessed Dataset/test-data-NT.pkl']

for i, save_path in enumerate(save_paths):
    text_path, concept_path = paths[i][0], paths[i][1]
    documents = getSentences(text_path, concept_path)
    # save documents as a dictionary to export into a pickle object
    documents_to_save = []
    for document in documents:
        documents_to_save.append(
            {
                'doc_name': document.doc_name,
                'sentences': document.sentences,
                'words': document.words,
                'concepts': document.concepts,
                'NLTK_POS': document.NLTK_POS,
                'GENIA_POS': document.GENIA_POS,
                'GENIA_chunks': document.GENIA_chunks,
                'GENIA_entities': document.GENIA_entities,
                'UMLS_sem': document.UMLS_sem,
                'UMLS_cui': document.UMLS_cui,
                'informative': document.informative
            }
        )

    with open(save_path, 'wb') as f:
        pickle.dump(documents_to_save, f)

