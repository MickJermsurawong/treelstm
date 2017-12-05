import tf_data_utils as utils

import sys
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

import tf_tree_lstm
import nary_tree_lstm

DIR = 'data/sst/'
GLOVE_DIR ='data/glove/'

import time

#from tf_data_utils import extract_tree_data,load_sentiment_treebank

class Config(object):

    num_emb=None

    emb_dim = 300
    hidden_dim = 150
    output_dim=None
    degree = 2
    num_labels = 3
    num_epochs = 20

    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=True
    nonroot_labels=True

    embeddings = None
    ancestral=False

    plot=False

    # ["PARENT", "SIBLING", "ALL"]
    span_scheme = "ALL"
    # ["DOT_PROD", "ADDITIVE", "MLP"]
    matching_scheme = "MLP"
    # ["ALL", "ROOT", "NONE"]
    attn_place = "ALL"

def train2(config): 
    import collections
    import numpy as np
    from sklearn import metrics

    def test(model, data, session, examine_attn=None):
        if config.fine_grained:
            relevant_labels = [0, 1, 2, 3, 4]
        else:
            relevant_labels = [0, 2]

        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = model.get_output()
            attn = model.get_attention()


            if config.fine_grained:
                y_true = batch[0].root_labels
            else:
                y_true = batch[0].root_labels/2

            feed_dict = {model.labels: batch[0].root_labels}
            feed_dict.update(model.tree_lstm.get_feed_dict(batch[0]))
            y_pred_, attn_ = session.run([y_pred, attn], feed_dict=feed_dict)
            y_pred_ = np.argmax(y_pred_[:,relevant_labels], axis=1)

            # display results

            if examine_attn and config.attn_place != 'NONE':
                batch_size = len(batch[0].sentences)
                root_attn = attn_[-batch_size:]
                chosen_sentence = [0, np.random.randint(1, batch_size)]
                examine_attn(batch[0].sentences, batch[0].sentence_lengths, root_attn, (y_true, y_pred_), chosen_sentence)

            ys_true += y_true.tolist()
            ys_pred += y_pred_.tolist()
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print "Accuracy", score
        #print "Recall", metrics.recall_score(ys_true, ys_pred)
        #print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print "confusion_matrix"
        print metrics.confusion_matrix(ys_true, ys_pred)
        return score

    def visualize_attn(sentence_word_ids, lengths, attention, ground_truth_and_preds, vocab, index=None):

        assert len(lengths) == len(sentence_word_ids)
        assert len(sentence_word_ids) == len(attention)

        if index is None:
            index = np.random.randint(0, len(sentence_word_ids), 2)

        chosen_batch_w = [[vocab.decode(c_id) for c_id in sen[:l]] for (sen, l) in zip(sentence_word_ids[index], lengths[index])]
        chosen_batch_a = attention[index]

        ground_truth, predicted = ground_truth_and_preds[0], ground_truth_and_preds[1]
        ground_truth, predicted = ground_truth[index], predicted[index]

        for (chosen_w, chosen_a, g, p) in zip(chosen_batch_w, chosen_batch_a, ground_truth, predicted):
            print("{} | G/P: {}/{}".format(g == p, g, p))
            for (w, a) in zip (chosen_w, chosen_a):
                print("{:<15} {}".format(w, a))
            print("------")

    data, vocab = utils.load_sentiment_treebank(DIR, GLOVE_DIR, config.fine_grained)
   # data, vocab = utils.load_sentiment_treebank(DIR, None, config.fine_grained)
    config.embeddings = vocab.embed_matrix

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    config.num_labels = num_labels
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb = num_emb
    config.output_dim = num_labels

    exp_name = experiment_name(config)
    print("----Start experiment {}".format(exp_name))

    # The colors of the different iterations
    plot_color = ['r', 'g', 'b']

    # Record the best test scores for each iteration
    best_test_score_array = []

    # Overlay 3 plots into the subplots
    for i in range(2):
        print("Experiment run# {}".format(i))
        train_set, dev_set, test_set = data['train'], data['dev'], data['test']

        # return
        random.seed()
        np.random.seed()

        from random import shuffle
        shuffle(train_set)
        train_set = utils.build_labelized_batch_trees(train_set, config.batch_size)
        dev_set = utils.build_labelized_batch_trees(dev_set, 500)
        test_set = utils.build_labelized_batch_trees(test_set, 500)

        with tf.Graph().as_default():

            #model = tf_seq_lstm.tf_seqLSTM(config)
            model = nary_tree_lstm.SoftMaxNarytreeLSTM(config)

            init=tf.global_variables_initializer()
            best_valid_score=0.0
            best_valid_epoch=0
            best_test_score=0.0
            dev_score=0.0
            test_score=0.0

            dev_score_array = []
            test_score_array = []
            loss_array = []

            with tf.Session() as sess:

                sess.run(init)

                for epoch in range(config.num_epochs):
                    start_time = time.time()
                    print 'epoch', epoch
                    avg_loss=0.0
                    avg_loss = model.train_epoch(train_set,sess)
                    loss_array.append(avg_loss)

                    print "Training time per epoch is {0}".format(
                        time.time() - start_time)

                    print 'validation score'
                    score = test(model, dev_set, sess, lambda sen, len, attn, g_p, idxs: visualize_attn(sen, len, attn, g_p, vocab, idxs))
                    # score = test(model, dev_set, sess)
                    dev_score_array.append(score)
                    #print 'train score'
                    #test(model, train_set[:40], sess)
                    if score >= best_valid_score:
                        best_valid_score = score
                        best_valid_epoch = epoch
                        test_score = test(model,test_set,sess)
                        best_test_score = test_score
                        test_score_array.append(test_score)
                    else:
                        test_score = test(model,test_set,sess)
                        test_score_array.append(test_score)
                    print 'Best test score :', best_test_score, 'updated', epoch - best_valid_epoch, 'epochs ago with validation score', best_valid_score

                    if config.plot:
                        plt.subplot(1,3,1)
                        plt.plot(range(epoch+1), loss_array, color=plot_color[i])
                        plt.ylabel("Average Loss")
                        plt.xlabel("Epochs")

                        plt.subplot(1, 3, 2)
                        plt.plot(range(epoch+1), dev_score_array, color=plot_color[i])
                        plt.ylabel("Dev Score")
                        plt.xlabel("Epochs")

                        plt.subplot(1, 3, 3)
                        plt.plot(range(epoch+1), test_score_array, color=plot_color[i])
                        plt.ylabel("Test Score")
                        plt.xlabel("Epochs")

                        plt.tight_layout()

                        if config.ancestral:
                            if config.fine_grained:
                                plt.savefig("Ancestral-Optimized-Fine-Grained.png")
                            else:
                                plt.savefig("Ancestral-Optimized-Binary.png")
                        else:
                            if config.fine_grained:
                                plt.savefig("Optimized-Fine-Grained-{}.png".format(exp_name))
                            else:
                                plt.savefig("Optimized-Binary-{}.png".format(exp_name))
        
        # Append best test scores
        best_test_score_array.append(best_test_score)

        # Overlay the subplots
        plt.hold(True)

    # Print out best test scores for all iterations
    print "Best test scores: "
    print best_test_score_array
    print("----End experiment {}".format(exp_name))


def experiment_name(config):

    return "atn-{}_mtching-{}_spn-{}_{}".format(
        config.attn_place,
        config.matching_scheme,
        config.span_scheme,
        time.strftime("%d-%m-%Y_%H-%M-%S")
    )


def train(restore=False):

    config=Config()
    config.batch_size = 5
    config.lr = 0.05
    config.fine_grained = True
    data,vocab = utils.load_sentiment_treebank(DIR,GLOVE_DIR,config.fine_grained)
    config.embeddings = vocab.embed_matrix
    config.early_stopping = 2
    config.reg = 0.0001
    config.dropout = 1.0
    config.emb_lr = 0.1
    config.ancestral = False
    config.plot = True

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    config.num_labels = num_labels
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb=num_emb
    config.output_dim = num_labels

    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)

    print config.maxnodesize,config.maxseqlen ," maxsize"
    #return 
    random.seed()
    np.random.seed()


    with tf.Graph().as_default():

        #model = tf_seq_lstm.tf_seqLSTM(config)
        model = tf_tree_lstm.tf_NarytreeLSTM(config)

        init=tf.global_variables_initializer()
        saver = tf.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0

        dev_score_array = []
        test_score_array = []
        loss_array = []

        with tf.Session() as sess:

            sess.run(init)


            if restore:saver.restore(sess,'./ckpt/tree_rnn_weights')
            for epoch in range(config.num_epochs):
                start_time = time.time()
                print 'epoch', epoch
                avg_loss=0.0
                avg_loss = train_epoch(model, train_set,sess)
                loss_array.append(avg_loss)
                print 'avg loss', avg_loss

                print "Training time per epoch is {0}".format(
                    time.time() - start_time)


                dev_score=evaluate(model,dev_set,sess)
                dev_score_array.append(dev_score)
                print 'dev-score', dev_score

                if dev_score >= best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    #saver.save(sess,'./ckpt/tree_rnn_weights')
                    test_score = evaluate(model, test_set, sess)
                    test_score_array.append(test_score)
                    print 'test score :', test_score, 'updated', epoch - best_valid_epoch, 'epochs ago with validation score', best_valid_score
                else:
                    test_score = evaluate(model, test_set, sess)
                    test_score_array.append(test_score)

                if config.plot:
                    plt.subplot(1,3,1)
                    plt.plot(range(epoch+1), loss_array)
                    plt.ylabel("Average Loss")
                    plt.xlabel("Epochs")

                    plt.subplot(1, 3, 2)
                    plt.plot(range(epoch+1), dev_score_array)
                    plt.ylabel("Dev Score")
                    plt.xlabel("Epochs")

                    plt.subplot(1, 3, 3)
                    plt.plot(range(epoch+1), test_score_array)
                    plt.ylabel("Test Score")
                    plt.xlabel("Epochs")

                    if config.ancestral:
                        if config.fine_grained:
                            plt.savefig("Ancestral-Non-Optimized-Fine-Grained.png")
                        else:
                            plt.savefig("Ancestral-Non-Optimized-Binary.png")
                    else:
                        if config.fine_grained:
                            plt.savefig("Non-Optimized-Fine-Grained.png")
                        else:
                            plt.savefig("Non-Optimized-Binary.png")

def train_epoch(model,data,sess):
    loss=model.train(data,sess)

    return loss

def evaluate(model,data,sess):
    acc=model.evaluate(data,sess)
    return acc

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--span', type=str, default='PARENT')
    parser.add_argument('--matching', type=str, default='DOT_PROD')
    parser.add_argument('--attn', type=str, default='ROOT')
    parser.add_argument('--ancs', type=bool, default=False)
    args = parser.parse_args()

    config = Config()
    config.batch_size = 25
    config.lr = 0.05
    config.dropout = 0.5
    config.reg = 0.0001
    config.emb_lr = 0.02
    config.fine_grained = True
    config.plot = True

    config.span_scheme = args.span
    config.matching_scheme = args.matching
    config.attn_place = args.attn
    config.ancestral = args.ancs

    train2(config)