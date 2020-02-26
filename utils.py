from data_structure import *
import re
import torch
from options import opt
import torch.autograd as autograd
import numpy as np
import nltk
import logging


pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

def text_tokenize_and_postagging(txt, sent_start):
    tokens= my_tokenize(txt)
    pos_tags = nltk.pos_tag(tokens)
    offset = 0
    for token, pos_tag in pos_tags:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start, pos_tag
        offset += len(token)

def token_from_sent(txt, sent_start):
    return [token for token in text_tokenize_and_postagging(txt, sent_start)]

def featureCapital(word):
    if word[0].isalpha() and word[0].isupper():
        return "1"
    else:
        return "0"

def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False

def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def wordToDigit(str):
    postStr =""
    wordToDigitMap = {'i':'1',  'ii':'2',  'iii':'3', 'iv':'4','v':'5', 'vi':'6'}
    strSplits = str.split(" ")
    for i in range(len(strSplits)):
        tempStr = strSplits[i]
        if i<len(strSplits)-1:
            if tempStr in wordToDigitMap.keys():
                postStr += wordToDigitMap[tempStr]+" "
            else:
                postStr += tempStr +" "
        else:
            if tempStr in wordToDigitMap.keys():
                postStr += wordToDigitMap[tempStr]
            else:
                postStr += tempStr

    return postStr

def clean_str(str):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    str = str.strip().lower()
    str = str.replace("/", " ")
    str = str.replace("-", " ")
    str = str.replace("(", " ( ")
    str = str.replace(")", " ) ")
    str = str.replace("+", " ")
    str = str.replace(",", "")
    str = re.sub("\'s", " \'s", str)
    str = re.sub(r"[^A-Za-z0-9]", " ", str)
    str = re.sub(r"\s{2,}", " ", str)
    return str.strip()


def output2Txt(outPath, outEntities, idx, isDev):
    
    if isDev:
        fPath = outPath + "/" + "dev" + "/"
    else:
        fPath = outPath + "/" + "test" + "/"
            
    fPath = fPath + str(idx)
    f = open(fPath,'w')
    
    for entity in outEntities:
        f.write(entity.docId + "\t"+ str(entity.start) + "\t"+ str(entity.end) + "\t" + \
                 entity.text + "\t" + 'Disease' + "\t" + entity.meshId + "\n")

    f.close()


def initEntity(entity, meshId):
    
    answerEntity = Entity()
    answerEntity.docId = entity.docId
    answerEntity.type = entity.type
    answerEntity.start = entity.start
    answerEntity.text = entity.text
    answerEntity.end = entity.end
    answerEntity.meshId = meshId
    
    return answerEntity

def findCandidates(example, dict):
    mwordslist = example.m_words
    candidate_concepts = list()
    for concept in dict.concepts:
        bPartSame = False
        for name in concept.words:
            set_same = set(mwordslist) & set(name)
            if len(set_same)>0:
                bPartSame = True
                break
                    
        if bPartSame:
            candidate_concepts.append(concept)
                    
    return candidate_concepts

def parser_dict(dict):
    words = list()
    label_to_ix = {}

    for id in dict.id_to_names.keys():
        names = dict.id_to_names[id]
        for name in names:
            for word in name.split(" "):
                if word not in words and word != '':
                    words.append(word)

    labels = set([label for label in dict.id_to_names.keys()])
    for label in labels:
        label_to_ix[label] = len(label_to_ix) + 1  # 0 is for unknown
    label_to_ix['-1'] = 0
    return labels, label_to_ix, words

def parser_corpus(traindocuments, devdocuments, testdocuments):
    words = []
    for doc in traindocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)

    for doc in devdocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)
    for doc in testdocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)
    return words

def parser_corpus_sent(traindocuments, devdocuments, testdocuments):
    words = []
    for doc in traindocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)
        for sent in doc.sents:
            for token in sent:
                if token['text'] not in words and token['text']!= '':
                    words.append(token['text'])


    for doc in devdocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)
            for sent in doc.sents:
                for token in sent:
                    if token['text'] not in words and token['text'] != '':
                        words.append(token['text'])
    for doc in testdocuments:
        for entity in doc.entities:
            entitywords = entity.text.split(" ")
            for entityword in entitywords:
                if entityword not in words and entityword !='':
                    words.append(entityword)
            for sent in doc.sents:
                for token in sent:
                    if token['text'] not in words and token['text'] != '':
                        words.append(token['text'])
    return words

def pad_sequence(x, max_len, eos_idx):
    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, 'EOS in sequence {row}'
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)

    return padded_x

def get_var(tensor, require_grad=False):
    if opt.use_cuda:
        tensor = tensor.cuda(opt.gpu)
    # return autograd.Variable(tensor, require_grad)
    return tensor


def sorted_collate(batch):
    return my_collate(batch, sort=True)

def unsorted_collate(batch):
    return my_collate(batch, sort=False)

def my_collate(batch, sort):
    x, y = zip(*batch)
    x1 = [s['entity'] for s in x]
    x2 = [s['feature'] for s in x]
    x3 = [s['sentence'] for s in x]
    x4 = [s['char_idx'] for s in x]

    x1, x2, x3, x4, y = pad(x1, x2, x3, x4, y, opt.pad_idx, sort)

    if torch.cuda.is_available():
        x1 = (x1[0].cuda(opt.gpu), x1[1].cuda(opt.gpu),x1[2].cuda(opt.gpu))
        x3 = (x3[0].cuda(opt.gpu), x3[1].cuda(opt.gpu))
        x4 = (x4[0].cuda(opt.gpu), x4[1].cuda(opt.gpu), x4[2].cuda(opt.gpu))
        y = y.cuda(opt.gpu)
    return (x1, x2, x3, x4, y)

def pad(x1, x2, x3, x4, y, eos_idx, sort):

    batch_size = len(x1)
    entity_lengths = [len(row) for row in x1]
    max_entity_len = max(entity_lengths)

    # entity
    padded_x = pad_sequence(x1, max_entity_len, eos_idx)
    entity_lengths = torch.LongTensor(entity_lengths)

    #sentence
    sentence_lengths = [len(row) for row in x3]
    max_sentence_len = max(sentence_lengths)
    padded_x3 = pad_sequence(x3, max_sentence_len, eos_idx)
    sentence_lengths = torch.LongTensor(sentence_lengths)

    ### deal with char
    chars = [char for char in x4]
    pad_chars = [chars[idx] + [[0]] * (max_entity_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_entity_len, max_word_len), dtype=torch.long))
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_recover = None
    entity_seq_recover = None

    y = torch.LongTensor(y).view(-1)
    if sort:
        if opt.sent_lstm:
            sentence_lengths, sort_idx = sentence_lengths.sort(0, descending=True)
            padded_x3 = padded_x3.index_select(0, sort_idx)

            y = y.index_select(0, sort_idx)
            x2 = [x2[idx] for idx in sort_idx]

            padded_x = padded_x.index_select(0, sort_idx)
            sort_len = entity_lengths.index_select(0, sort_idx)

            char_seq_tensor = char_seq_tensor[sort_idx].view(batch_size * max_entity_len, -1)
            char_seq_lengths = char_seq_lengths[sort_idx].view(batch_size * max_entity_len, )
            char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
            char_seq_tensor = char_seq_tensor[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)

            _, entity_seq_recover = sort_idx.sort(0, descending=False)
            return (padded_x, sort_len, entity_seq_recover), x2, (padded_x3, sentence_lengths), (
            char_seq_tensor, char_seq_lengths, char_seq_recover), y

        else:            
            sort_len, sort_idx = entity_lengths.sort(0, descending=True)
            padded_x = padded_x.index_select(0, sort_idx)
            y = y.index_select(0, sort_idx)

            x2 =[x2[idx] for idx in sort_idx]

            padded_x3 = padded_x3.index_select(0,sort_idx)
            sentence_lengths = sentence_lengths.index_select(0,sort_idx)
            char_seq_tensor = char_seq_tensor[sort_idx].view(batch_size * max_entity_len, -1)
            char_seq_lengths = char_seq_lengths[sort_idx].view(batch_size * max_entity_len, )
            char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
            char_seq_tensor = char_seq_tensor[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)
            _, entity_seq_recover = sort_idx.sort(0, descending=False)
            return (padded_x, sort_len, entity_seq_recover), x2, (padded_x3,sentence_lengths), (char_seq_tensor,char_seq_lengths,char_seq_recover), y
    else:
        return (padded_x, entity_lengths, entity_seq_recover), x2, (padded_x3,sentence_lengths), (char_seq_tensor, char_seq_lengths, char_seq_recover), y


def endless_get_next_batch(loaders, iters):
    try:
        inputs, features, sentences, chars, targets = next(iters)
    except StopIteration:
        iters = iter(loaders)
        inputs, features, sentences, chars, targets = next(iters)

    return (inputs, features, sentences, chars, targets)

def generate_word_alphabet(corpus_words, dict_words):
    word_to_ix = {}
    words = list(set(corpus_words) | set(dict_words))

    char_to_ix = {}
    char_to_ix["<unk>"] = len(char_to_ix)
    for word in words:
        if word not in word_to_ix.keys():
            word_to_ix[word] = len(word_to_ix)
        for char in word:
            if char not in char_to_ix.keys():
                char_to_ix[char] = len(char_to_ix)

    return word_to_ix, words,char_to_ix


def removeStrInParentheses(str):
    new_str = ""
    idleftPa = str.find('(')
    idrightPa = str.find(')')
    if idleftPa != -1 and idrightPa != -1:
        leftstr = str[0:idleftPa].strip()
        rightstr = str[idrightPa + 1:].strip()
        new_str += leftstr + " " + rightstr
    else:
        new_str = str

    return new_str.strip()

def calculateMacroAveragedFMeasure(test_instances, testdocuments):

    correctAnnotations = prepareCorrectAnnotattions(testdocuments)
    foundAnnotations = getFoundAnnotations(test_instances)

    correctAnnotationsDict = dict()
    for correctAnnotation in correctAnnotations:
        docId = correctAnnotation.getDocId()
        if docId in correctAnnotationsDict.keys():
            docCorrectAnnotations= correctAnnotationsDict[docId]
            if correctAnnotation not in docCorrectAnnotations:
                docCorrectAnnotations.append(correctAnnotation)
        else:
            docCorrectAnnotations = []
            docCorrectAnnotations.append(correctAnnotation)
            correctAnnotationsDict[docId] = docCorrectAnnotations

    foundAnnotationsDict = dict()
    for foundAnnotation in foundAnnotations:
        docId = foundAnnotation.getDocId()
        if docId in foundAnnotationsDict.keys():
            docFoundAnnotations = foundAnnotationsDict[docId]
            if foundAnnotation not in docFoundAnnotations:
                docFoundAnnotations.append(foundAnnotation)
        else:
            docFoundAnnotations = []
            docFoundAnnotations.append(foundAnnotation)
            foundAnnotationsDict[docId] = docFoundAnnotations


    keys = correctAnnotationsDict.keys()
    keys = keys|foundAnnotationsDict.keys()

    pSum,rSum,fSum = 0.0,0.0,0.0
    for key in keys:
        ca = correctAnnotationsDict[key]
        if not ca:
            ca = set()

        fa = foundAnnotationsDict[key]
        if not fa:
            fa =set()

        tpSet = [a for a in ca if a in fa] # intersection
        tp = len(tpSet)
        fp = len(fa)-tp
        fn = len(ca)-tp
        p_1 = r_1 = 1.0
        if tp +fp !=0:
            p_ = 1.0*tp/(tp+fp)
        if tp+fn != 0:
            r_ = 1.0*tp/(tp+fn)
        f_=0.0
        if p_+r_ > 0:
            f_ = 2*p_*r_/(p_+r_)
        pSum += p_
        rSum += r_
        fSum += f_

    p = pSum/len(keys)
    r = rSum / len(keys)
    f = fSum / len(keys)

    return p,r,f

def prepareCorrectAnnotattions(documents):
    correctAnnatations = list()
    for doc in documents:
        for entity in doc.entities:
            currentAnnotation = AbstractAnnotation(entity.doc_id, entity.gold_meshId)
            if currentAnnotation not in correctAnnatations:
                correctAnnatations.append(currentAnnotation)
    return correctAnnatations

def getFoundAnnotations(instances):
    foundAnnotations = list()
    for instance in instances:
        foundAnnotation = AbstractAnnotation(instance[0],instance[5])
        if foundAnnotation not in foundAnnotations:
            foundAnnotations.append(foundAnnotation)
    return foundAnnotations

def get_sentences_and_tokens_from_nltk(text, nlp_tool, entities):
    all_sents_inds = []
    generator = nlp_tool.span_tokenize(text)
    for t in generator:
        all_sents_inds.append(t)

    sentences = []
    for ind in range(len(all_sents_inds)):
        t_start = all_sents_inds[ind][0]
        t_end = all_sents_inds[ind][1]

        tmp_tokens = token_from_sent(text[t_start:t_end], t_start)
        sentence_tokens = []
        for token_idx, token in enumerate(tmp_tokens):
            token_dict = {}
            token_dict['start'], token_dict['end'] = token[1], token[2]
            token_dict['text'] = token[0]
            token_dict['pos'] = token[3]
            token_dict['cap'] = featureCapital(token[0])
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
        
    for entity in entities:

        for sentence in sentences:
            if entity.start>= int(sentence[0]['start']) and entity.end <= int(sentence[-1]['end']):
                entity_sentence = []
                for token_dict in sentence:
                    entity_sentence.append(token_dict)
                entity.sentence = entity_sentence
                break
    return sentences


