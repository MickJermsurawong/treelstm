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
    num_epochs = 30


    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=True
    nonroot_labels=True

    embeddings = None
    ancestral=False

    plot=False

def train2():

    config = Config()
    config.batch_size = 25
    config.lr = 0.05
    config.dropout = 0.5
    config.reg = 0.0001
    config.emb_lr = 0.02
    config.fine_grained = True
    config.plot = True


    import collections
    import numpy as np
    from sklearn import metrics

    def test(model, data, session):
        if config.fine_grained:
            relevant_labels = [0, 1, 2, 3, 4]
        else:
            relevant_labels = [0, 2]

        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = model.get_output()

            if config.fine_grained:
                y_true = batch[0].root_labels
            else:
                y_true = batch[0].root_labels/2

            feed_dict = {model.labels: batch[0].root_labels}
            feed_dict.update(model.tree_lstm.get_feed_dict(batch[0]))
            y_pred_ = session.run([y_pred], feed_dict=feed_dict)
            y_pred_ = np.argmax(y_pred_[0][:,relevant_labels], axis=1)
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

    # The colors of the different iterations
    plot_color = ['r', 'g', 'b']

    # Overlay 3 plots into the subplots
    for i in range(3):
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
                    avg_loss = model.train_epoch(train_set[:],sess)
                    loss_array.append(avg_loss)

                    print "Training time per epoch is {0}".format(
                        time.time() - start_time)

                    print 'validation score'
                    score = test(model,dev_set,sess)
                    dev_score_array.append(score)
                    #print 'train score'
                    #test(model, train_set[:40], sess)
                    if score >= best_valid_score:
                        best_valid_score = score
                        best_valid_epoch = epoch
                        test_score = test(model,test_set,sess)
                        test_score_array.append(test_score)
                    else:
                        test_score = test(model,test_set,sess)
                        test_score_array.append(test_score)
                    print 'test score :', test_score, 'updated', epoch - best_valid_epoch, 'epochs ago with validation score', best_valid_score

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
                                plt.savefig("Optimized-Fine-Grained.png")
                            else:
                                plt.savefig("Optimized-Binary.png")

        # Overlay the subplots
        plt.hold(True)

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
    if len(sys.argv) > 1:
        if(sys.argv[1] == "-optimized"):
            print "running optimized version"
            train2()
        else:
            print "running not optimized version"
            train()
    else:
        print "running not optimized version, run with option -optimized for the optimized one"
        train()

