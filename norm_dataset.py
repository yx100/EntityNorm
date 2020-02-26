from torch.utils.data import Dataset

class NormDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return(self.X[idx], self.Y[idx])

def getNormInstance(documents, vocab, label_to_ix,char_to_ix):
    X = []
    Y = []
    for doc in documents:
        for entity in doc.entities:
            entity_words = []
            entity_features =[]
            sentence_words = []
            char_idx = []
            instance = {'entity': entity_words, 'feature':entity_features, 'sentence':sentence_words,'char_idx':char_idx}
            for entity_word in entity.text.split():
                if entity_word != '':
                    entity_words.append(vocab.lookup(entity_word))
                    token_char = []
                    for char in entity_word:
                        token_char.append(char_to_ix[char] if char in char_to_ix else char_to_ix["<unk>"])
                    char_idx.append(token_char)

            if len(entity_words) == 0:
                continue
            instance['entity'] = entity_words

            #feature for output
            entity_features.append(entity.doc_id)
            entity_features.append(str(entity.start))
            entity_features.append(str(entity.end))
            entity_features.append(entity.text)
            entity_features.append("Disease")
            entity_features.append("preID")
            instance['feature'] = entity_features

            #sentence for context information
            if entity.sentence is not None:
                sentence_words = [token_dict['text'] for token_dict in entity.sentence if token_dict['text']!='']
            else:
                print('wrong entity')
            sentence_words = [vocab.lookup(word) for word in sentence_words]
            instance['sentence'] = sentence_words
            instance['char_idx'] = char_idx


            X.append(instance)

            if entity.gold_meshId in label_to_ix.keys():
                Y.append(label_to_ix[entity.gold_meshId])
            else:
                print('not id=', entity.gold_meshId)
                ab = label_to_ix['-1']
                Y.append(label_to_ix['-1'])


    set = NormDataset(X, Y)
    return set

def getDictInstance(dict, vocab, label_to_ix, char_to_ix):
    X = []
    Y = []
    for id, names in dict.id_to_names.items():
        for name in names:

            name_words = []
            features = []
            sentence_words = []
            char_idx = []

            instance = {'entity': name_words, 'feature':features,'sentence':sentence_words,'char_idx':char_idx}
            for word in name.split(" "):
                if word !='':
                    name_words.append(word)
                    token_char = []
                    for char in word:
                        token_char.append(char_to_ix[char] if char in char_to_ix else char_to_ix["<unk>"])
                    char_idx.append(token_char)

            if len(name_words) == 0:
                continue

            instance['entity'] = [vocab.lookup(word) for word in name_words]

            sentence_words = [vocab.lookup('concept')]
            instance['feature'] = features
            instance['sentence'] = sentence_words

            instance['char_idx'] = char_idx
            X.append(instance)
            if id in label_to_ix.keys():
                Y.append(label_to_ix[id])

    set = NormDataset(X, Y)
    return set
