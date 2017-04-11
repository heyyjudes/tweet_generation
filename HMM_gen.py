import numpy as np
import re
from hmmlearn import hmm
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def build_vec_dict(input_list):
    vocab_dict = {}
    reverse_vocab_dict = {}
    index = 0
    for twt in input_list:
        twt = twt.rstrip("\n")
        tokens = twt.split(" ")
        for t in tokens:
            if t not in vocab_dict.keys():
                vocab_dict[t] = index
                reverse_vocab_dict[index] = t
                index += 1

    return vocab_dict, reverse_vocab_dict

def build_tweet_vec(vocab_dict, tweet, lengths):
    size = len(vocab_dict.keys())
    tweet = tweet.rstrip("\n")
    tokens = tweet.split(" ")
    sent_arr = []
    lengths.append(len(tokens))
    for t in tokens:
        num = vocab_dict[t]
        word_arr = np.zeros((size,), dtype=np.int)
        word_arr[num] = 1
        sent_arr.append(word_arr)
    return np.asarray(sent_arr)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

    with open("data/obama_tweet_nopad.txt") as f:
        my_tweets = f.readlines()

    vocab_dict, reverse_vocab_dict = build_vec_dict(my_tweets)

    lengths = []
    train_vecs = np.concatenate([build_tweet_vec(vocab_dict, z, lengths) for z in my_tweets])

    train_vecs = train_vecs[:3200]
    lengths = lengths[:100]

    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=20, covariance_type="diag")
    x_1 = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
    x_2 = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
    x_new = np.concatenate([x_1, x_2])
    lengths_mini = [len(x_1), len(x_2)]
    print x_new.shape
    print train_vecs.shape

    model.fit(train_vecs, lengths)
    print np.argmax(model._generate_sample_from_state(0)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(0))]
    print np.argmax(model._generate_sample_from_state(1)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(1))]
    print np.argmax(model._generate_sample_from_state(2)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(2))]
    print np.argmax(model._generate_sample_from_state(3)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(3))]
    print np.argmax(model._generate_sample_from_state(4)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(4))]
    print np.argmax(model._generate_sample_from_state(5)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(5))]
    print np.argmax(model._generate_sample_from_state(6)), reverse_vocab_dict[np.argmax(model._generate_sample_from_state(6))]

