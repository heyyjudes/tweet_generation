import markovify
import re

with open("data/obama_tweet_1000.txt") as f:
    text = f.read()

# Build the model.
text_model = markovify.NewlineText(text, state_size=1)

# Print three randomly-generated sentences of no more than 140 characters
for i in range(3):
    sentence = text_model.make_short_sentence(140)
    new_sentence = re.sub(r"<[a-z]+>", '', sentence)
    print new_sentence
