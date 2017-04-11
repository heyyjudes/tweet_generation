import numpy as np
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
    tweet = tweet.rstrip("\n")
    tokens = tweet.split(" ")
    sent_arr = []

    for t in tokens:
        if not t.startswith("<"):
            num = vocab_dict[t]
            word_arr = np.zeros((size,), dtype=np.int)
            word_arr[num] = 1
            sent_arr.append(word_arr)

    lengths.append(len(sent_arr))
    return np.asarray(sent_arr)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

    with open("data/obama_tweet_1000.txt") as f:
        my_tweets = f.readlines()

    vocab_dict, reverse_vocab_dict = build_vec_dict(my_tweets)

    lengths = []
    train_vecs = np.concatenate([build_tweet_vec(vocab_dict, z, lengths) for z in my_tweets])
    print train_vecs.shape
    train_vecs = train_vecs[:9900]
    lengths = lengths[:300]
    num_comp = 17
    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=num_comp, covariance_type='diag')
    model.transmat_ = np.random.rand(num_comp, num_comp)

    model.fit(train_vecs, lengths)
    print model.transmat_
    print model.transmat_.shape
    print model._generate_sample_from_state(0)
    print model._generate_sample_from_state(3)

    for i in range(0, num_comp):
        state_arr = model._generate_sample_from_state(i)
        print np.argmax(state_arr), reverse_vocab_dict[np.argmax(state_arr)]

