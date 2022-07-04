import nltk


# Tokenlization: nltk.word_tokenlization
sentence = """
At eight o'clock on Thursday morning
Arthur didn't feel very good.
"""

tokens = nltk.word_tokenize(sentence)
for token in tokens:
    print(tokens)

# Add tags as a parser
taggs = nltk.pos_tag(tokens)
tagged[0:6]