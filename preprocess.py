import csv
import numpy as np
import HTMLParser
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import *

error_list = []
def twtt1(input_str):
    ''' this function uses regular expression to remove html tags and attributes'''
    clean_txt = re.compile(r'<[^>]+>')
    new_str = clean_txt.sub('', input_str)
    return new_str

def twtt2(input_str):
    '''this function uses HTMLParser to change html character codes to ascii'''
    h = HTMLParser.HTMLParser()
    new_str = ""
    try:
        new_str = h.unescape(input_str)
    except UnicodeDecodeError:
        error_list.append(input_str)
        print "decode error"
    return new_str

def twtt3(input_str):
    '''this function removes URLS by splitting string and looking for URL beginnings'''

    new_str = re.sub(r'https?:\/\/.*[\r\n]*', '', input_str)
    new_str = re.sub(r'www?:\/\/.*[\r\n]*', '', new_str)
    new_str = re.sub(r'http?:\/\/.*[\r\n]*', '', new_str)
    new_str = re.sub(r'[#\w.]+twitter.com[\/\w]+', '', new_str)

    return new_str

def twtt4(input_str, token_dict, max_len):
    '''remove usernames, hashtags and punctuation at the first character of user names'''

    new_str = re.sub(r"[^A-Za-z]+", ' ', input_str)

    tokens = TreebankWordTokenizer().tokenize(new_str.lower())

    if max_len < len(tokens):
        max_len = len(tokens)
    token_dict += tokens

    return tokens, max_len

def twttpad(tokens, max_size):

    length = len(tokens) + 2

    tokens.append('<end>')
    tokens.insert(0, '<beg>')
    while length < max_size:
        tokens.append('<pad>')
        length += 1

    return " ".join(tokens)

def load_tokens():
    token_arr = []
    with open('data/unlabeled_tweet.txt', 'rb') as f:

        lines = f.readlines()
        for row in lines:
            tokens = row.split(" ")
            token_arr.append(tokens)

    return token_arr



if __name__ == "__main__":
    # generate output tweet file with punctuation removed: 1 tweet per line
    input_path = "data/BarackObama.csv"
    output_file = "data/obama_tweet_1000.txt"
    #change this number for number of tweets to include
    num_tweets = 500


    my_tweets = []
    with open(input_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            my_tweets.append(row)

    #for sentiment tweets
    # pos = my_tweets[:num_tweets]
    # neg = my_tweets[800000:800000+num_tweets]
    # new_tweets = pos + neg

    #for obama and trump tweets

    tweet_indx = np.random.choice(len(my_tweets) - 1, 1000)
    new_tweets = np.asarray(my_tweets)[tweet_indx]


    max_len = 31
    token_dict = []
    out_f = open(output_file, 'w')
    for row in new_tweets:
        final_str = twtt1(row[5])
        final_str = twtt2(final_str)
        final_str = twtt3(final_str)
        tokens, max_len = twtt4(final_str, token_dict, max_len)
        final_str = twttpad(tokens, max_len)
        final_str = final_str.lstrip(" ")
        out_f.write(final_str + '\n')

    token_arr = load_tokens()
    print "total", len(token_dict)
    print "set", len(set(token_dict))
    print max_len

