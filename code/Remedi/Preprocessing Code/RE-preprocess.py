import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

import os
import re  # for regular expression extraction
import pickle

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

# to temp store information of relations in a relation file - for easier pre-processing
class Relation:
    def __init__(self, e1, e2, rel, lineNo, e1start, e2start, e1end, e2end):
        self.e1 = e1
        self.e2 = e2
        self.rel = rel
        self.lineNo = lineNo
        self.e1start = e1start
        self.e2start = e2start
        self.e1end = e1end
        self.e2end = e2end

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
        self.relations = []
        self.e1_dist = []
        self.e2_dist = []
        self.informative = []  # is the sentence informative or non-informative (has no tagged NER)

    # adding one sentence and its features to the document
    def update(self, sentence, words, concepts, NLTK_POS, GENIA_POS, GENIA_chunks, GENIA_entities, UMLS_sem, UMLS_cui, relations, e1_dist, e2_dist, informative):
        self.sentences.append(sentence)
        self.words.append(words)
        self.concepts.append(concepts)
        self.NLTK_POS.append(NLTK_POS)
        self.GENIA_POS.append(GENIA_POS)
        self.GENIA_chunks.append(GENIA_chunks)
        self.GENIA_entities.append(GENIA_entities)
        self.UMLS_sem.append(UMLS_sem)
        self.UMLS_cui.append(UMLS_cui)
        self.relations.append(relations)
        self.e1_dist.append(e1_dist)
        self.e2_dist. append(e2_dist)
        self.informative.append(informative)

# return document with the same filename
def getDocument(data, filename):
    for document in data:
        if document['doc_name'] == filename:
            return document

def getRelations(relations, lineNo):
    final = []
    for relation in relations:
        if relation.lineNo == lineNo:
            final.append(relation)
    return final

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

# determine the relation of an entity pair
def findRelation(entity_pair, relations):
    current_relation = 'NPPR'
    e1, e2 = entity_pair[0], entity_pair[1]
    if e1.con == 'treatment':
        current_relation = 'NTPR'

    for relation in relations:
        # check if the words match by the entities positioning
        if e1.wordStart == relation.e1start and e2.wordStart == relation.e2start:
            current_relation = relation.rel
    
    return current_relation

def parseRelations(relSentences):
    final = [] # to return
    for sentence in relSentences:
        # get the word, line number, word start and word end and tag
        frags = sentence.strip('\n').split("||")
        relation = re.search('"(.*)"', frags[1]).group(1) 
        entity1 = re.search('"(.*)"', frags[0]).group(1)
        entity2 = re.search('"(.*)"', frags[2]).group(1)

        # to get the line number and word number
        entity1temp = frags[0].replace('c=\"'+entity1+'\"', '').strip(' ').split(' ')
        entity2temp = frags[2].replace('c=\"'+entity2+'\"', '').strip(' ').split(' ')
        # get the line numbers
        e1line = int(entity1temp[0].split(':')[0])
        e2line = int(entity2temp[0].split(':')[0])                   

        # Assumptions - ignore non P-T or P-P problems
        if relation == "TeCP" or relation == "TeRP":
            continue
        if e1line != e2line: # IMPORTANT dont consider relations on multiple sentences right now
            continue
        
        e1wordstart = int(entity1temp[0].split(':')[1])
        e1wordend = int(entity1temp[1].split(':')[1])
        e2wordstart = int(entity2temp[0].split(':')[1])
        e2wordend = int(entity2temp[1].split(':')[1])

        final.append(Relation(entity1, entity2, relation, e1line - 1, e1wordstart, e2wordstart, e1wordend, e1wordend))
    return final

# input = textPath, conceptPath
def getSentences(textP, relP, data):
    documents = []  # to store a list of Document objects
    for r, d, f in os.walk(textP):
        for txtfile in f:
            if txtfile.endswith(".txt"):
                filename = os.path.splitext(txtfile)[0]  # get name of file without extension
                document = Document(filename)  # create a new document object
                current_data = getDocument(data, filename)

                textPath = textP + filename + '.txt'
                relPath = relP + filename + '.rel'

                # open files and get all sentences
                with open(textPath) as f:
                    textSentences = f.readlines()
                with open(relPath) as f:
                    relSentences = f.readlines()

                # parse concept file into list of objects
                relations = parseRelations(relSentences)

                # for each sentence in the text file
                for lineNo, textSentence in enumerate(textSentences):
                    # get the words and labels and POS
                    sentence = textSentence.strip('\n')
                    # get the words for the current line number
                    words = current_data['words'][lineNo]

                    # get all features for the current line number
                    concepts = current_data['concepts'][lineNo]
                    NLTK_POS = current_data['NLTK_POS'][lineNo]
                    GENIA_POS = current_data['GENIA_POS'][lineNo]
                    GENIA_chunks = current_data['GENIA_chunks'][lineNo]
                    GENIA_entities = current_data['GENIA_entities'][lineNo]
                    UMLS_sem = current_data['UMLS_sem'][lineNo]
                    UMLS_cui = current_data['UMLS_cui'][lineNo]

                    entities = getEntities(concepts, words, lineNo)
                    entity_pairs = getEntityPairs(entities)
                    current_relations = getRelations(relations, lineNo)

                    if(len(entity_pairs) == 0): # skip sentences with no entity pairs
                        continue

                    for entity_pair in entity_pairs:
                        # get the distance between words and entities
                        e1_dist, e2_dist = getDistance(entity_pair, words)
                        # check if a relation exists for the entity pairs
                        relation = findRelation(entity_pair, current_relations)
                        # determine if the sentence is informative or non informative
                        informative = True
                        if relation == "NPPR" or relation == "NTPR": 
                            informative = False
                        document.update(sentence, words, concepts, NLTK_POS, GENIA_POS, GENIA_chunks, GENIA_entities, UMLS_sem, UMLS_cui, relation, e1_dist, e2_dist, informative)

                documents.append(document)
    return documents

data_paths = [ '../Preprocessed Dataset/beth-data-NT.pkl', '../Preprocessed Dataset/partners-data-NT.pkl', '../Preprocessed Dataset/test-data-NT.pkl']
save_paths = [ '../Preprocessed Dataset/beth-data-NT-RE.pkl', '../Preprocessed Dataset/partners-data-NT-RE.pkl', '../Preprocessed Dataset/test-data-NT-RE.pkl']

for i, data_path in enumerate(data_paths):
    text_path, concept_path = paths[i][0], paths[i][2]
    with open(data_path, 'rb') as f: data = pickle.load(f) # load pre-processed data
    documents = getSentences(text_path, concept_path, data)

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
                'relations': document.relations,
                'e1_dist': document.e1_dist,
                'e2_dist': document.e2_dist,
                'UMLS_cui': document.UMLS_cui,
                'informative': document.informative
            }
        )
        
    save_path = save_paths[i]
    with open(save_path, 'wb') as f:
        pickle.dump(documents_to_save, f)