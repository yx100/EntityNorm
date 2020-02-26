import torch
import torch.nn as nn  ## neural net library
import torch.optim as optim  # optimization package

import utils
from data_helpers import loadAbbreviations,load_dict,preprocessMentions,parserCdrTxtFile
from options import opt
import random
import numpy as np
from vocab import Vocab
from model.mentionatten_charatten import AttenATTEN
from torch.utils.data import DataLoader
import norm_dataset
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info(opt)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

def evaluate(data_loader, model, label_to_ix):

    ix_to_label = {v:k for k,v in label_to_ix.items()}
    correct = 0
    total = 0
    model.eval()
    loader_it = iter(data_loader)
    num_it = len(data_loader)
    instances = []
    for j in range(num_it):
        mention_inputs, features, sentences,char_inputs, targets = utils.endless_get_next_batch(data_loader, loader_it)
        targets = utils.get_var(targets)
        pred = model(mention_inputs, char_inputs)
        _, y_pred = torch.max(pred, 1)
        total += targets.size(0)
        # correct += (y_pred == targets).sum().sample_data[0]
        correct += (y_pred == targets).sum().item()
        # output evaluate
        pred_numpy = (y_pred.data).cpu().numpy()
        y_pred_labels = [ix_to_label[ix] for ix in pred_numpy]
        assert len(y_pred_labels)==len(features), 'y_pred_labels and features have different lengths'
        for i, pred_label in enumerate(y_pred_labels):
            features[i][5] = pred_label
            instances.append(features[i])

    acc = 100.0 * correct / total

    return acc, instances

if __name__ == "__main__":

    traindocuments = parserCdrTxtFile(opt.train_file)
    # traindocuments = parserCdrTxtFile(opt.trainDev_file)
    devdocuments = parserCdrTxtFile(opt.dev_file)
    testdocuments = parserCdrTxtFile(opt.test_file)

    entityAbbres = loadAbbreviations(opt.abbre_file)
    preprocessMentions(traindocuments, devdocuments, testdocuments, entityAbbres)
    dict = load_dict(opt.dict_file)
    meshlabels, meshlabel_to_ix, dict_words = utils.parser_dict(dict)

    corpus_words = utils.parser_corpus(traindocuments, devdocuments, testdocuments)
    word_to_ix, all_words, char_to_ix = utils.generate_word_alphabet(corpus_words, dict_words)

    if opt.random_emb:
        opt.emb_filename = ''
    vocab = Vocab(word_to_ix, opt.emb_filename, opt.word_emb_size)

    dict_instances = norm_dataset.getDictInstance(dict, vocab, meshlabel_to_ix,char_to_ix)
    train_instances = norm_dataset.getNormInstance(traindocuments,vocab, meshlabel_to_ix,char_to_ix )
    dev_instances = norm_dataset.getNormInstance(devdocuments, vocab, meshlabel_to_ix,char_to_ix)
    test_instances = norm_dataset.getNormInstance(testdocuments, vocab, meshlabel_to_ix,char_to_ix)

    logging.info('dict_instances_len {}'.format(len(dict_instances)))
    logging.info('train_instance_len {}'.format(len(train_instances)))

    my_collate = utils.sorted_collate

    dict_loader = DataLoader(dict_instances, opt.batch_size, shuffle=True, collate_fn=my_collate)
    train_loader = DataLoader(train_instances, opt.batch_size, shuffle=True, collate_fn = my_collate)
    dev_loader = DataLoader(dev_instances, opt.batch_size, shuffle=False, collate_fn = my_collate)
    test_loader = DataLoader(test_instances, opt.batch_size, shuffle=False, collate_fn = my_collate)

    train_iter = iter(train_loader)
    num_iter = len(train_loader)

    model = AttenATTEN(vocab, len(meshlabel_to_ix),char_to_ix)

    if torch.cuda.is_available():
        model.cuda(opt.gpu)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if opt.fine_tune:
        utils.unfreeze_net(model.embedding)
    else:
        utils.freeze_net(model.embedding)
    #pre-training dictionary instance
    if opt.pretraining:

        dict_iter = iter(dict_loader)
        dict_num_iter = len(dict_loader)

        #start training dictionary
        logging.info("batch_size: %s,  dict_num_iter %s, train num_iter %s" % (str(opt.batch_size), str(dict_num_iter), str(num_iter)))
        for epoch in range(opt.dict_iteration):
            epoch_start = time.time()

            # sum_dict_cost = 0.0
            correct_1, total_1 = 0, 0

            model.train()

            for i in range(dict_num_iter):

                dict_inputs, _, dict_sentences, dict_char_inputs, dict_targets = utils.endless_get_next_batch(dict_loader, dict_iter)

                dict_batch_output = model(dict_inputs,dict_char_inputs)
                dict_cost = criterion(dict_batch_output, dict_targets)
                # sum_dict_cost += dict_cost.item()

                # for dict training accuracy
                total_1 += len(dict_inputs[1])
                _, dict_pred = torch.max(dict_batch_output, 1)

                correct_1 += (dict_pred == dict_targets).sum().item()


                dict_cost.backward()
                optimizer.step()
                model.zero_grad()

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            # logging.info("Epoch {}, training time {}, dict Loss {:.4f}".format((epoch+1), epoch_cost, sum_dict_cost/(dict_num_iter + 1)))

            logging.info('Epoch {}, Dict Training Accuary: {:.2f}%'.format((epoch+1), 100.0 * correct_1 / total_1))
            if 100.0*correct_1/total_1 == 100.0:
                break


    ##train corpus
    for epoch in range(opt.max_epoch):
        model.train()
        optimizer.zero_grad()
        # sum_cost = 0.0
        correct, total = 0, 0

        for i in range(num_iter):
            mention_inputs, _, sentences, char_inputs, targets = utils.endless_get_next_batch(train_loader, train_iter)

            batch_output = model(mention_inputs, char_inputs)
            cost = criterion(batch_output, targets)

            # sum_cost += cost.item()

            # train accuracy
            total += len(mention_inputs[1])
            _, pred = torch.max(batch_output, 1)
            # correct += (pred == targets).sum().sample_data[0]
            correct += (pred == targets).sum().item()

            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

        logging.info('Epoch {}, Training Accuary: {:.2f}%'.format((epoch + 1), 100.0 * correct / total))

        # evaluate on test sample_data
        test_acc, test_instances = evaluate(test_loader, model, meshlabel_to_ix)
        logging.info('Epoch {}, Testing Accuary: {:.2f}%'.format((epoch + 1), test_acc))

        p, r, f = utils.calculateMacroAveragedFMeasure(test_instances, testdocuments)

        logging.info('Epoch {}, Macro P= {:.4f}, R= {:.4f}, F= {:.4f}'.format((epoch + 1), p, r, f))

    print('norm end')
