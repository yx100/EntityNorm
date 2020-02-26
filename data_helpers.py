import codecs
from data_structure import *
from utils import clean_str,wordToDigit,get_sentences_and_tokens_from_nltk
from options import opt
import logging
import nltk

def load_dict(path):
    dict = Dictionary()
    concepts =[]
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue

        concept = Concept()

        linesplit = line.split("\t")
        concept.meshId = linesplit[0].strip()

        name_synonym = list()
        for idx in range(1, len(linesplit)):
            names = linesplit[idx]

            names = clean_str(names)
            if opt.use_word2digit:
                names = wordToDigit(names)
            if names == '':
                continue
            name_synonym.append(names)
        concept.set_names(name_synonym)
        concepts.append(concept)

    dict.set_concepts(concepts)
    dict.set_id_to_names()
    return dict

def loadAbbreviations(abbrePath):
    abbreviations = list()
    lines = codecs.open(abbrePath, 'r', 'utf-8')
    for line in lines:
        line = line.strip().lower()
        if line=='':
            continue
        linesplits = line.split("\t")
        abbre = DiseaseAbbreviation()

        if len(linesplits) < 3:
            print(line)
        linesplits[1] = clean_str(linesplits[1])
        linesplits[2] = clean_str(linesplits[2])
        if opt.use_word2digit:
            linesplits[1] = wordToDigit(linesplits[1])
            linesplits[2] = wordToDigit(linesplits[2])

        abbre.initAbbre(linesplits[0].strip(), linesplits[1], linesplits[2])
        if abbre not in abbreviations:
            abbreviations.append(abbre)
    return abbreviations

def preprocessMentions(traindocuments, devdocuments, testdocuments, abbreviations):
    # abbreviation replace
    for doc in traindocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break

    for doc in devdocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break

    for doc in testdocuments:
        for entity in doc.entities:
            for abbre in abbreviations:
                if doc.doc_name == abbre.docId:
                    if entity.text == abbre.sf:
                        entity.text = abbre.lf
                        break
def parserCdrTxtFile(path):
    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')

    documents = []
    id=title=abstractt = ""
    doc = Document()
    with codecs.open(path, 'r', 'utf-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                _t_position = line.find('|t|')
                if _t_position != -1:
                    id = line[0: _t_position]
                    title = line[_t_position + len('|t|'):]

                _a_position = line.find('|a|')
                if _a_position != -1:
                    abstractt = line[_a_position + len('|a|'):]

                lineSplits = line.split("\t")
                if len(lineSplits)>5:
                    if lineSplits[4] =="Disease":
                        lineSplits[3] = clean_str(lineSplits[3])
                        if len(lineSplits) == 6:
                            entity = Entity()

                            if lineSplits[5] == "-1":
                                continue

                            if lineSplits[5].find("|") != -1:
                                meshId = lineSplits[5].strip().split("|")[1]
                                entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]),lineSplits[3], lineSplits[4], meshId)
                            else:
                                entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]), lineSplits[3], lineSplits[4], lineSplits[5].strip())
                            doc.entities.append(entity)
                        elif len(lineSplits) == 7:
                            entity = Entity()
                            entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]), lineSplits[3],lineSplits[4], lineSplits[5].strip())
                            entity.subEntiyText = lineSplits[6]
                            doc.entities.append(entity)
            else:
                if len(id)>0 and len(title)>0 and len(abstractt)>0:
                    doc.doc_name = id
                    doc.title = title
                    doc.abstractt = abstractt
                    document_text = title + " " + abstractt
                    sentences = get_sentences_and_tokens_from_nltk(document_text.lower(), nlp_tool, doc.entities)
                    doc.sents = sentences
                    documents.append(doc)
                    id = title = abstractt = ""
                    doc = Document()

        if len(id)>0 and len(title)>0 and len(abstractt)>0:
            doc.name = id
            doc.title = title
            doc.abstractt = abstractt
            document_text = title + " " + abstractt
            sentences = get_sentences_and_tokens_from_nltk(document_text.lower(), nlp_tool, doc.entities)
            doc.sents = sentences
            documents.append(doc)

    # Decompose a composite entity into simple entities
    entitycount = 0
    ret_documents = []
    for doc in documents:
        ret_doc = Document()
        ret_doc.sents = doc.sents
        ret_doc.initDocument(doc.doc_name, doc.title, doc.abstractt)
        for entity in doc.entities:

            idx = entity.gold_meshId.find("|")
            # composite entity
            if (idx != -1):
                simpleEntities = generateEntities(entity)
                ret_doc.entities.extend(simpleEntities)
                entitycount += len(simpleEntities)
            else:
                # simple entity
                ret_doc.entities.append(entity)
                entitycount+=1

        ret_documents.append(ret_doc)
    logging.info('entity count= {}'.format(entitycount))
    return ret_documents

def generateEntities(entity):

    ret_entities = []
    start = entity.start
    end = entity.end
    meshIds = entity.gold_meshId.split("|")
    entityTexts = entity.subEntiyText.split("|")
    assert len(meshIds) == len(entityTexts),"error entity_3 {}, text {}".format(entity.doc_id, entity.text)

    for i in range(len(meshIds)):
        subEntity = Entity()
        entityTexts[i] = clean_str(entityTexts[i])
        if i==0:
            leftEnd = start + len(entityTexts[i])
            subEntity.subEntityInitWithEntity(start, leftEnd, entityTexts[i], meshIds[i], entity)
        elif i==len(meshIds)-1:
            rightStart = end - len(entityTexts[i])
            subEntity.subEntityInitWithEntity(rightStart, end, entityTexts[i], meshIds[i], entity)
        else:
            firstWord = entityTexts[i].split(" ")[0]
            middleStart = start + entity.text.find(firstWord)
            middleEnd = middleStart + len(entityTexts[i])
            subEntity.subEntityInitWithEntity(middleStart, middleEnd, entityTexts[i], meshIds[i], entity)

        ret_entities.append(subEntity)

    return ret_entities

def parserCdrTxtFile_simple(path,total_id_list, total_entity_str_list):
    if opt.nlp_tool == "nltk":
        nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')

    documents = []
    id_list = []
    entity_list = []
    entity_str_list = []
    id=title=abstractt = ""
    doc = Document()
    with codecs.open(path, 'r', 'utf-8') as fp:
        compositecount = 0
        for line in fp:
            line = line.strip()
            if line != '':
                _t_position = line.find('|t|')
                if _t_position != -1:
                    id = line[0: _t_position]
                    title = line[_t_position + len('|t|'):]

                _a_position = line.find('|a|')
                if _a_position != -1:
                    abstractt = line[_a_position + len('|a|'):]

                lineSplits = line.split("\t")
                if len(lineSplits) > 5:
                    if lineSplits[5] != "-1":
                        if lineSplits[5].lower().strip() not in id_list:
                            id_list.append(lineSplits[5].lower().strip())
                        if lineSplits[5].lower().strip() not in total_id_list:
                            total_id_list.append(lineSplits[5].lower().strip())
                        if lineSplits[3].lower().strip() not in entity_str_list:
                            entity_str_list.append(lineSplits[3].lower().strip())
                        if lineSplits[3].lower().strip() not in total_entity_str_list:
                            total_entity_str_list.append(lineSplits[3].lower().strip())




                    if lineSplits[4] == "Disease":
                        lineSplits[3] = clean_str(lineSplits[3])
                        entity = Entity()
                        entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]),
                                                         lineSplits[3], lineSplits[4], lineSplits[5].strip())
                        doc.entities.append(entity)
                        entity_list.append(entity)


                # if len(lineSplits)>5:
                #     if lineSplits[4] =="Disease":
                #         lineSplits[3] = clean_str(lineSplits[3])
                #         if len(lineSplits) == 6:
                #             entity = Entity()
				#
                #             if lineSplits[5].find("|") != -1:
                #                 meshId = lineSplits[5].strip().split("|")[1]
                #                 entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]),lineSplits[3], lineSplits[4], meshId)
                #             else:
                #                 entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]), lineSplits[3], lineSplits[4], lineSplits[5].strip())
                #             doc.entities.append(entity)
                #         elif len(lineSplits) == 7:
                #             compositecount+=1
                #             continue
                #             entity = Entity()
                #             entity.setEntity(lineSplits[0], int(lineSplits[1]), int(lineSplits[2]), lineSplits[3],lineSplits[4], lineSplits[5].strip())
                #             entity.subEntiyText = lineSplits[6]
                #             doc.entities.append(entity)
            else:
                if len(id)>0 and len(title)>0 and len(abstractt)>0:
                    doc.doc_name = id
                    doc.title = title
                    doc.abstractt = abstractt
                    document_text = title + " " + abstractt
                    sentences = get_sentences_and_tokens_from_nltk(document_text.lower(), nlp_tool, doc.entities)
                    doc.sents = sentences
                    documents.append(doc)
                    id = title = abstractt = ""
                    doc = Document()

    # Decompose a composite entity into simple entities
    # entitycount = 0
    # ret_documents = []
    # for doc in documents:
    #     ret_doc = Document()
    #     ret_doc.sents = doc.sents
    #     ret_doc.initDocument(doc.doc_name, doc.title, doc.abstractt)
    #     for entity in doc.entities:
	#
    #         idx = entity.gold_meshId.find("|")
    #         # composite entity
    #         if (idx != -1):
    #             simpleEntities = generateEntities(entity)
    #             ret_doc.entities.extend(simpleEntities)
    #             entitycount += len(simpleEntities)
    #         else:
    #             # simple entity
    #             ret_doc.entities.append(entity)
    #             entitycount += 1
	#
    #     ret_documents.append(ret_doc)
    # logging.info('test simple entity count= {}'.format(entitycount))
    # return ret_documents
    return len(entity_list), len(id_list), len(entity_str_list)


def readwrongresult(wrongfile):
    entities = []
    for line in codecs.open(wrongfile, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        linesplits = line.split("\t")
        if len(linesplits) == 6:
            entity = Entity()
            entity.doc_id = linesplits[0].strip()
            entity.start = int(linesplits[1])
            entity.end = int(linesplits[2])
            entity.text = linesplits[3].strip()
            entity.gold_meshId = linesplits[5].strip().split(" ")[1]
            entities.append(entity)


    return entities