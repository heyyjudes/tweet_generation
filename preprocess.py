import csv
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
    arr = input_str.split(" ")
    for token in arr:
        if token.startswith("www") or token.startswith("http"):
            arr.remove(token)
    new_str = " ".join(arr)
    return new_str

def twtt4(input_str, token_dict, max_len):
    '''remove usernames, hashtags and punctuation at the first character of user names'''
    clean_txt = re.compile(r'#')
    new_str = clean_txt.sub('', input_str)
    clean_txt = re.compile(r'@[\w]+')
    new_str = clean_txt.sub('', new_str)

    clean_txt = re.compile(r'\'')
    new_str = clean_txt.sub('', new_str)

    clean_txt = re.compile(r"[-.,!?;*%&<:()'\"\\]+")
    new_str = clean_txt.sub(' ', new_str)

    clean_txt = re.compile(r"\d+")
    new_str = clean_txt.sub('', new_str)

    tokens = TreebankWordTokenizer().tokenize(new_str.lower())
    new_str = " ".join(tokens)
    clean_txt = re.compile(r'\\[\w]+')
    new_str = clean_txt.sub('', new_str)
    tokens = new_str.split(" ")
    if max_len < len(tokens):
        max_len = len(tokens)
        print tokens
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
    input_path = "data/tweets.csv"
    output_file = "data/unlabeled_tweet.txt"
    #change this number for number of tweets to include
    num_tweets = 500


    my_tweets = []
    with open(input_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            my_tweets.append(row)

    pos = my_tweets[:num_tweets]
    neg = my_tweets[800000:800000+num_tweets]
    new_tweets = pos + neg

    max_len = 31
    token_dict = []
    out_f = open(output_file, 'w')
    for row in new_tweets:
        final_str = twtt1(row[5])
        final_str = twtt2(final_str)
        final_str = twtt3(final_str)
        tokens, max_len = twtt4(final_str, token_dict, max_len)

        final_str = twttpad(tokens, 31)
        final_str = final_str.lstrip(" ")
        out_f.write(final_str + '\n')

    token_arr = load_tokens()
    print "total", len(token_dict)
    print "set", len(set(token_dict))
    print max_len

