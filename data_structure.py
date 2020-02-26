
class Token():
    def __init__(self):
        self.word = ""
        self.begin = -1
        self.end = -1
        self.sentId = ""
        self.pos = ""
        self.lemma = ""
        self.label = ""

    def setToken(self, word, begin, end):
        self.word = word
        self.begin = begin
        self.end = end
class Sent:
    def __init__(self):
        self.tokens = []
        self.begin = -1
        self.end = -1
        self.docId = ""
        self.sentId = ""
        
    def setSent(self, tokens, begin, end, docId, sentId):
        self.tokens = list(tokens) 
        self.begin = begin
        self.end = end 
        self.docId = docId
        self.sentId = sentId
        
class Entity:
    def __init__(self):
        self.start = -1
        self.text = ""
        self.end = -1
        self.doc_id = ""
        self.gold_meshId = ""
        self.pre_meshId = ""
        self.sentence = None
        self.subEntiyText = ''

    def setEntity(self, docId, start, end, text, type, meshId):
        self.doc_id = docId
        self.start = start
        self.end = end
        self.text = text
        self.type = type
        self.gold_meshId = meshId
        
    def equalsBoundary(self, another):
        if (self.start == another.start and self.text == another.text and self.doc_id == another.doc_id):
            return True
        else:
            return False
    def initialiEntity(self, entity):
        self.start = entity.start
        self.end = entity.end
        self.doc_id= entity.doc_id
        self.gold_meshId = entity.gold_meshId
    def subEntityInitWithEntity(self, start, end, text, meshId, entity):
        self.start = start
        self.end = end
        self.text = text
        self.gold_meshId = meshId
        self.doc_id = entity.doc_id
        self.type = entity.type
        self.sentence = entity.sentence
        
class Document:
    def __init__(self):
        self.doc_name = ""
        self.entities = []
        self.sents = None
        self.conceptIds = []
        self.title = ""
        self.abstractt = ""
    def initDocument(self, id, title, abstractt):
        self.doc_name = id
        self.title = title
        self.abstractt = abstractt
    def getConceptIds(self):

        for entity in self.entities:
            if entity.gold_meshId not in self.conceptIds:
                self.conceptIds.append(entity.gold_meshId)

        return self.conceptIds
    def getDocName(self):
        return self.doc_name

        
class Concept:
    def __init__(self):
        self.meshId = ""
        self.names = []
    def set_concept(self, id, names):
        self.meshId = id
        self.names = names
    def set_names(self, nameslist):
        for name in nameslist:
            if name not in self.names:
                self.names.append(name)

class OutEntity:
    def __init__(self):
        self.documentId = ""
        self.type =  "Disease"
        self.start = -1
        self.end = -1
        self.text = ""
        self.mesh = "-1"
    
    def equals(self, another):
        if(self.start == another.start and self.documentId == another.documentId):
            return True
        else:
            return False
        
class Dictionary:
    def __init__(self):
        self.id_to_names = {}
        self.concepts = []
        self.alternateIDMap = {}
    def set_concepts(self, conceptslist):
        for concept in conceptslist:
            if concept not in self.concepts:
                self.concepts.append(concept)
    def set_id_to_names(self):
        for concept in self.concepts:
            self.id_to_names[concept.meshId] = concept.names
         
class DiseaseAbbreviation:
    def __init__(self): 
        self.docId = ""
        self.sf = ""
        self.lf=""  
    def initAbbre(self, id, sf, lf):
        self.docId = id
        self.sf = sf
        self.lf = lf

class AbstractAnnotation:
    def __init__(self, docId, conceptId):
        self.docId = docId
        self.conceptId= conceptId
    def getDocId(self):
        return self.docId
    def getConceptId(self):
        return self.conceptId

    def __eq__(self, other):
        if self.docId != other.docId:
            return False
        if self.conceptId != other.conceptId:
            return False
        return True