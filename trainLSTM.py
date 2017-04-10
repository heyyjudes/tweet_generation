import os
import numpy as np
import pickle
import logging
import argparse
import time
import tensorflow as tf
import math
from tensorflow.contrib.tensorboard.plugins import projector
import LSTM_simple

timestr = time.strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument(
'--eps',
type=float,
default=0.01,
help='Initial learning rate.'
)
parser.add_argument(
'--batch_size',
type=int,
default=100,
help='Batch Size'
)
parser.add_argument(
'--num_epochs',
type=int,
default=100,
help='Total number of epochs'
)
parser.add_argument(
'--nlp_dim',
type=int,
default=300,
help='dimension of word embedding vector'
)
parser.add_argument(
'--summaries_dir',
type=str,
default="./logs/%s/"%(timestr),
help='Summaries/Tensorboard Location'
)
parser.add_argument(
'--checkpoint',
type=str,
default=None,
help='checpoint file'
)
parser.add_argument(
'--mode',
type=str,
default="trainTrump",
help='operation mode'
)
parser.add_argument(
'--num_words',
type=int,
default=30,
help='Number of words to be generated'
)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        summary_list.append(tf.summary.scalar('mean', mean))
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    summary_list.append(tf.summary.scalar('stddev', stddev))
    summary_list.append(tf.summary.scalar('max', tf.reduce_max(var)))
    summary_list.append(tf.summary.scalar('min', tf.reduce_min(var)))
    summary_list.append(tf.summary.histogram('histogram', var))


def log_register(filename, typ, name):
    """Register a new file in the catalog
    Args:
        filename: string, path to the log file.
        typ: string, file type, "csv" or "plain" or "image".
        name: string, name of the visualization.
    """
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    catalog = os.path.join(folder, 'catalog')
    basename = os.path.basename(filename)
    if not os.path.exists(catalog):
        with open(catalog, 'w') as f:
            f.write('filename,type,name\n')
            f.write('{},{},{}\n'.format(basename, typ, name))
    else:
        with open(catalog, 'a') as f:
            f.write('{},{},{}\n'.format(basename, typ, name))


def setup_logger(logger_name, log_file, level=logging.INFO, stream=False, writeFile=True):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter()
    l.setLevel(level)
    if writeFile==True:
        if not os.path.exists(logger_name):
            mode = 'w'
        else:
            mode = 'a'
        fileHandler = logging.FileHandler(log_file, mode=mode)
        fileHandler.setFormatter(formatter)
        l.addHandler(fileHandler)

    if stream==True:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)


def init_weight(fan_in, fan_out, name=None, stddev=1.0):
    #Initialize with Xavier initialization
    weights = tf.Variable(tf.random_normal([fan_in, fan_out], stddev=stddev/math.sqrt(float(fan_in))), name=name)
    return weights

def init_bias(fan_out, name=None, stddev=1.0):
    #Initialize with zero
    bias = tf.Variable(tf.zeros([fan_out]), name=name)
    return bias


def init_word_embedding(word_count, vocab_size=300, oneHot=True):
    with tf.name_scope('embeddings'):
        if oneHot == True:
            initial = tf.eye(word_count)
            print(initial)
            embedding = tf.Variable(initial, trainable=False)
        else:
            initial = tf.random_normal([word_count, vocab_size], stddev=1.0/math.sqrt(word_count))
            embedding = tf.Variable(initial)

    return embedding

def loadData(path):
    '''
    Loads the NLP data (word embeddings for Q&A) for all MSCOCO images.
    Returns:
        -dictionary {ids: features}
        -dictioanry {ids: images}
    '''
    train_data = np.load(path)
    return np.asarray(train_data['feature']), np.asarray(train_data['length'])


def generate(num_words, prompt='<beg>', dictIdxtoT=None, dictTtoIdx=None):
    """ Accepts a current character, initial state"""
    saver = tf.train.import_meta_graph(FLAGS.checkpoint+".meta")
    vocab_size = len(dictIdxtoT)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, FLAGS.checkpoint)
        graph = tf.get_default_graph()
        '''
        for n in tf.get_default_graph().as_graph_def().node:
            print n.name 
        '''
        x_tweets = graph.get_tensor_by_name('x_tweets/Placeholder:0')
        init_state = graph.get_tensor_by_name('state/zeros_1:0')
        predictions = graph.get_tensor_by_name('predictions/Softmax:0')
        last_states = graph.get_tensor_by_name('final_states/RNN/transpose:0')
        batch_size = graph.get_tensor_by_name('batch_size/Placeholder:0')
        length = graph.get_tensor_by_name('length/Placeholder:0')
        state = None
        current_word = dictTtoIdx[prompt]
        words = []

        for i in range(num_words):
            if state is not None:
                feed_dict={x_tweets: [[current_word]]*FLAGS.batch_size, init_state: state, batch_size:FLAGS.batch_size, length:[1]*FLAGS.batch_size}
            else:
                feed_dict={x_tweets: [[current_word]]*FLAGS.batch_size, batch_size:FLAGS.batch_size, length:[1]*FLAGS.batch_size}

            preds, state = sess.run([predictions,last_states], feed_dict)
            state = state.reshape((FLAGS.batch_size,-1))
            current_word = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            words.append(current_word)

    words = map(lambda x: dictIdxtoT[x], words)
    print(" ".join(words))
    return(" ".join(words))


def trainTrump():
    trainData, trainLength= loadData("data/trump_txt.npz")
    num_total = len(trainData)

    #Loading Dictionaries...
    dictIdxtoT = pickle.load(open("dictionaries/dictIdxtoT.pkl", "rb"))
    dictTtoIdx = pickle.load(open("dictionaries/dictTtoIdx.pkl", "rb"))

    vocab_size = len(dictIdxtoT)
    words_dim = trainData.shape[1]


    #Logging for the training statistics
    train_ce_list = []
    train_acc_list = []
    global summary_list
    summary_list = []

    if not os.path.isdir(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
    setup_logger('ce', FLAGS.summaries_dir+"/ce.csv")
    log_ce = logging.getLogger('ce')
    log_ce.info("step,time,train")
    log_register(FLAGS.summaries_dir+"/ce.csv", 'csv', 'Cross Entropy')
    setup_logger('acc', FLAGS.summaries_dir+"/acc.csv")
    log_acc = logging.getLogger('acc')
    log_acc.info("step,time,train")
    log_register(FLAGS.summaries_dir+"/acc.csv", 'csv', 'Accuracy')
    setup_logger('raw', FLAGS.summaries_dir+"/raw.log", stream=True)
    logger = logging.getLogger('raw')
    log_register(FLAGS.summaries_dir+"/raw.log", 'plain', 'Raw logs')
    setup_logger('metadata', FLAGS.summaries_dir+"/metadata.tsv")
    metadata = logging.getLogger('metadata')
    #Create metadata file for visualizing the embedding
    metadata.info("Name")
    for key in sorted(dictIdxtoT.keys()):
        metadata.info(dictIdxtoT[key])

    '''
    Declaration of our training variables
    '''
    nlp_dim = FLAGS.nlp_dim
    hidden_dim = 512
    n_lstm_steps = words_dim
    num_classes = len(dictTtoIdx)

    with tf.name_scope('Wemb'):
        Wemb = init_word_embedding(vocab_size, nlp_dim)
        variable_summaries(Wemb)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = os.path.join(FLAGS.summaries_dir, 'metadata.tsv')
    embedding.tensor_name = Wemb.name


    with tf.name_scope('x_tweets'):
        x_tweets = tf.placeholder(tf.int32, shape=[None, None])
    tEmb = tf.nn.embedding_lookup(Wemb, x_tweets)
    with tf.name_scope('y_'):
        y_ = tf.placeholder(tf.int32, shape=[None, None])
    with tf.name_scope('batch_size'):
        batch_size = tf.placeholder(dtype=tf.int32)
    with tf.name_scope('length'): 
        length = tf.placeholder(dtype=tf.int32, shape=[None])

    cell = LSTM_simple.CustomCell(hidden_dim, hidden_dim, input_keep_prob=1.0, output_keep_prob=1.0)
    with tf.name_scope('state'): 
        init_state = cell.zero_state(batch_size, tf.float32)

    with tf.name_scope('final_states'):
        rnn_outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            initial_state=init_state,
            inputs=tEmb,
            sequence_length=length)

    W = init_weight(hidden_dim, num_classes, name='W')
    b = init_bias(num_classes, name='b')
    rnn_outputs = tf.reshape(rnn_outputs, [-1, hidden_dim])
    y_flatten = tf.reshape(y_, [-1])
    logits = tf.matmul(rnn_outputs, W) + b
    with tf.name_scope('predictions'):
        predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_flatten))
    summary_list.append(tf.summary.scalar('total loss', total_loss))
    merged = tf.summary.merge(summary_list)
    train_step = tf.train.AdamOptimizer(FLAGS.eps).minimize(total_loss)

    saver = tf.train.Saver(max_to_keep=2)

    epoch_num = 0
    last_ce = 0
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                          sess.graph)
    
    projector.visualize_embeddings(train_writer, config)

    logger.info("Beginning Trainning...")
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while epoch_num < FLAGS.num_epochs:
            logger.info("Epoch Trained: %d/%d"%(epoch_num, FLAGS.num_epochs))
            saver.save(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"), global_step=epoch_num)
            train_shuffle = np.arange(len(trainData))
            np.random.shuffle(train_shuffle)
            trainData = trainData[train_shuffle]
            trainLength = trainLength[train_shuffle]

            i=0
            while i*FLAGS.batch_size<len(trainData):
                #Batching:
                start = i*FLAGS.batch_size
                end = min((i+1)*FLAGS.batch_size, len(trainData))

                train_batch_x = trainData[start:end]
                train_batch_y = trainData[start:end]
                train_batch_length = trainLength[start:end]

                train_summ, train_ce = sess.run([merged, total_loss], 
                            feed_dict={x_tweets: train_batch_x, 
                                       y_: train_batch_y,
                                       batch_size: len(train_batch_y),
                                       length: train_batch_length})

                sess.run(train_step, feed_dict={x_tweets: train_batch_x, 
                                       y_: train_batch_y,
                                       batch_size: len(train_batch_y),
                                       length: train_batch_length})

                i+=1
                logger.info("Train---Loss: %.4f... (%d/%d)" %(train_ce, end, len(trainData)))

            epoch_num+=1

def trainObama():
    trainData, trainLength= loadData("data/obama_txt.npz")
    num_total = len(trainData)

    #Loading Dictionaries...
    dictIdxtoT = pickle.load(open("dictionaries/dictIdxtoT.pkl", "rb"))
    dictTtoIdx = pickle.load(open("dictionaries/dictTtoIdx.pkl", "rb"))

    vocab_size = len(dictIdxtoT)
    words_dim = trainData.shape[1]

    if not os.path.isdir(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
    setup_logger('ce', FLAGS.summaries_dir+"/ce.csv")
    log_ce = logging.getLogger('ce')
    log_ce.info("step,time,train")
    log_register(FLAGS.summaries_dir+"/ce.csv", 'csv', 'Cross Entropy')
    setup_logger('acc', FLAGS.summaries_dir+"/acc.csv")
    log_acc = logging.getLogger('acc')
    log_acc.info("step,time,train")
    log_register(FLAGS.summaries_dir+"/acc.csv", 'csv', 'Accuracy')
    setup_logger('raw', FLAGS.summaries_dir+"/raw.log", stream=True)
    logger = logging.getLogger('raw')
    log_register(FLAGS.summaries_dir+"/raw.log", 'plain', 'Raw logs')
    setup_logger('metadata', FLAGS.summaries_dir+"/metadata.tsv")
    metadata = logging.getLogger('metadata')
    #Create metadata file for visualizing the embedding
    metadata.info("Name")
    for key in sorted(dictIdxtoT.keys()):
        metadata.info(dictIdxtoT[key])

    '''
    Declaration of our training variables
    '''
    nlp_dim = FLAGS.nlp_dim
    hidden_dim = 512
    n_lstm_steps = words_dim
    num_classes = len(dictTtoIdx)

    with tf.name_scope('Wemb'):
        Wemb = init_word_embedding(vocab_size, nlp_dim)
        variable_summaries(Wemb)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = os.path.join(FLAGS.summaries_dir, 'metadata.tsv')
    embedding.tensor_name = Wemb.name


    with tf.name_scope('x_tweets'):
        x_tweets = tf.placeholder(tf.int32, shape=[None, None])
    tEmb = tf.nn.embedding_lookup(Wemb, x_tweets)
    with tf.name_scope('y_'):
        y_ = tf.placeholder(tf.int32, shape=[None, None])
    with tf.name_scope('batch_size'):
        batch_size = tf.placeholder(dtype=tf.int32)
    with tf.name_scope('length'): 
        length = tf.placeholder(dtype=tf.int32, shape=[None])

    cell = LSTM_simple.CustomCell(hidden_dim, hidden_dim, input_keep_prob=1.0, output_keep_prob=1.0)
    with tf.name_scope('state'): 
        init_state = cell.zero_state(batch_size, tf.float32)

    with tf.name_scope('final_states'):
        rnn_outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            initial_state=init_state,
            inputs=tEmb,
            sequence_length=length)

    W = init_weight(hidden_dim, num_classes, name='W')
    b = init_bias(num_classes, name='b')
    rnn_outputs = tf.reshape(rnn_outputs, [-1, hidden_dim])
    y_flatten = tf.reshape(y_, [-1])
    logits = tf.matmul(rnn_outputs, W) + b
    with tf.name_scope('predictions'):
        predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_flatten))
    summary_list.append(tf.summary.scalar('total loss', total_loss))
    merged = tf.summary.merge(summary_list)
    train_step = tf.train.AdamOptimizer(FLAGS.eps).minimize(total_loss)

    saver = tf.train.Saver(max_to_keep=2)

    epoch_num = 0
    last_ce = 0
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                          sess.graph)
    
    projector.visualize_embeddings(train_writer, config)

    logger.info("Beginning Trainning...")
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while epoch_num < FLAGS.num_epochs:
            logger.info("Epoch Trained: %d/%d"%(epoch_num, FLAGS.num_epochs))
            saver.save(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"), global_step=epoch_num)
            train_shuffle = np.arange(len(trainData))
            np.random.shuffle(train_shuffle)
            trainData = trainData[train_shuffle]
            trainLength = trainLength[train_shuffle]

            i=0
            while i*FLAGS.batch_size<len(trainData):
                #Batching:
                start = i*FLAGS.batch_size
                end = min((i+1)*FLAGS.batch_size, len(trainData))

                train_batch_x = trainData[start:end]
                train_batch_y = trainData[start:end]
                train_batch_length = trainLength[start:end]

                train_summ, train_ce = sess.run([merged, total_loss], 
                            feed_dict={x_tweets: train_batch_x, 
                                       y_: train_batch_y,
                                       batch_size: len(train_batch_y),
                                       length: train_batch_length})

                i+=1
                logger.info("Train---Loss: %.4f... (%d/%d)" %(train_ce, end, len(trainData)))

            epoch_num+=1



if __name__ == '__main__':
    global FLAGS
    FLAGS = parser.parse_args()

    if FLAGS.mode=="trainTrump":
        trainTrump()
    if FLAGS.mode=="trainObama":
        trainObama()
    if FLAGS.mode=="generateTrump":
        dictIdxtoT = pickle.load(open("dictionaries/dictIdxtoT.pkl", "rb"))
        dictTtoIdx = pickle.load(open("dictionaries/dictTtoIdx.pkl", "rb"))
        generate(FLAGS.num_words, dictIdxtoT=dictIdxtoT, dictTtoIdx=dictTtoIdx)
    if FLAGS.mode=="generateObama":
        dictIdxtoT = pickle.load(open("dictionaries/dictIdxtoO.pkl", "rb"))
        dictTtoIdx = pickle.load(open("dictionaries/dictOtoIdx.pkl", "rb"))
        generate(FLAGS.num_words, dictIdxtoT=dictIdxtoT, dictTtoIdx=dictTtoIdx)

