import string
from nltk.corpus import stopwords
import pandas as pd
import re
import spacy

## get raw data extracted from *.sgm

df = pd.read_csv("/Users/becca/Desktop/dossierAlternanceXingyuLIU/data/reuters2.csv")

## helper function to clean tokens of Reuter in the corpus
def clean(s):
    s = s.replace('Reuter\n&#3;', '')
    return s

## dump the cleaned sentences into a new column
for i, row in df.iterrows():
    df.at[i, "body_clean"] = clean(row.body)

# --------------- general statistics -------------- #
# articles in total
print(f'total number of articles: {len(df)}')
# combination of topics = 13 (5!*2)
df['foc_topics'].nunique()
# number of articles per category
focused_topics = ['money-fx', 'ship', 'interest', 'acq', 'earn']
for tp in focused_topics:
    print(f'topic {tp} : {sum(df[tp])}')
# articles with multiple categories 689
mul_cat_number = len(df[df['multi_topics'] == 1])
print(f'articles with multiple categories: {mul_cat_number}')
# articles with multiple categories listed in focused_categories (5 categories of our job)
i_multi_focused = 0
new = []
for tp in df['foc_topics']:
    tp = re.sub(r'[\]\[\' ]','',tp)
    tp = tp.split(',')
    # print(tp)
    if len(tp) > 1:
        i_multi_focused += 1
        new.append(tp)
print(f'articles with multiple categories within our five topics : {i_multi_focused}')



# --------------- refactor the Reuters Corpus -------------- #

nlp = spacy.load('en_core_web_sm')

## construct stop words set
# add spacy stopwords
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
myStops = set(stopwords.words('english')) | {punct for punct in string.punctuation} | spacy_stopwords

# lexical diversity = unique tokens / tokens
def lexical_diversity(l_tokens):
    # numbers and spaces are considered as the same
    # concatenate the list with ,
    str = ",".join(l_tokens)
    # replace digits with digitt
    ## floats
    str = re.sub(r'\d+\.?\d+','digittt',str)
    ## single digit
    str = re.sub(r'\d','digittt',str)
    # replace numbers and spaces
    l_tokens = str.split(',')
    # true lexical diversity
    return len(set(l_tokens)) / len(l_tokens)

# average length of tokens

def avg_len_tokens(l_tokens):
    length_vector = [len(x) for x in l_tokens]
    return sum(length_vector) / len(length_vector)

# calculate average length of sentences
def avg_len_sent(l_sentences):
    length_vector = [len(x.split()) for x in l_sentences]
    return sum(length_vector) / len(l_sentences)

print("start dataframe construction")

# refactor the data frame
for i, row in df.iterrows():
    # sanity check the existence of the content of the body
    if (row["body_clean"]):
        # replace \n because it doesn't represent a line feed here, we can have more accurate sentence tokenization
        content = row["body_clean"].replace("\n"," ")
        doc = nlp(content)
        # tokens without punctuation
        no_punc_tokens = []
        ## collect lemmas and different parts of speech
        lemmas = []
        adjectives = []
        nouns = []
        verbs = []
        adverbs = []
        # ! no stopwords list
        no_stop_punct_tokens = []
        # collect sentences here
        sentences = []
        # collect named entity here
        people = []
        org = []
        quantity = []
        # tokenize sentences
        for sent in doc.sents:
            sentences.append(sent.string.strip())
        ## only collect wanted entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.append(ent.text)
            if ent.label_ == "ORG":
                org.append(ent.text)
            if ent.label_ == "QUANTITY":
                quantity.append(ent.text)
        ## collect basic lingustic infos
        for token in doc:
            ## save to variable to avoir repetitive retrieval
            ttoken = token.text.lower()
            tpos = token.pos_
            if ttoken not in string.punctuation and not ttoken.isspace():
                no_punc_tokens.append(ttoken)
                lemmas.append(token.lemma_.lower())
            if ttoken not in myStops and not ttoken.isspace():
                no_stop_punct_tokens.append(token.lemma_.lower())
            if tpos == "ADJ":
                adjectives.append(token.lemma_.lower())
            if tpos == "NOUN" or token.pos_ == "PROPN":
                nouns.append(token.lemma_.lower())
            if tpos == "VERB":
                verbs.append(token.lemma_.lower())
            if tpos == "ADV":
                adverbs.append(token.lemma_.lower())
        # these sentences are with normal \n
        df.at[i, "body_sentences"] = "\n".join(sentences)
        df.at[i, "body_token"] = " ".join(no_punc_tokens)
        df.at[i, "body_lemma"] = " ".join(lemmas)
        ##  For frequent Words Display
        df.at[i, "body_lemma_noStop"] = " ".join(no_stop_punct_tokens)
        df.at[i, "body_nouns"] = " ".join(nouns)
        df.at[i, "body_adjectives"] = " ".join(adjectives)
        df.at[i, "body_verbs"] = " ".join(verbs)
        df.at[i, "body_people"] = ",".join(people)
        df.at[i, "body_org"] = ",".join(org)
        df.at[i, "body_quantity"] = ",".join(quantity)
        df.at[i, "no_tokens"] = len(no_punc_tokens)
        df.at[i, "no_types"] = len(set(no_punc_tokens))
        df.at[i, "no_lemmas"] = len(lemmas)
        df.at[i, "avg_wordLength"] = avg_len_tokens(no_punc_tokens)
        df.at[i, "avg_sentLength"] = avg_len_sent(sentences)
        df.at[i,"lexical_diversity"] = lexical_diversity(no_punc_tokens)
        df.at[i,"lemma_diversity"] = lexical_diversity(lemmas)

print("end of dataframe construction")

# export data to csv
df.to_csv('reuters_wrangled.csv',index=False)
print("end of export")
print("API construction accomplished :)")




# ------------------------- extras ---------------- #

# For illustration purpose when presenting the report
# doc = nlp("He was going to play basketball.")
# for word in doc:
#     print(word.text)
#     if word.is_stop:
#         print(word)
# I want PEOPLE ORG QUANTITY
# https://spacy.io/api/annotation
# doc = nlp("Indians spent over $71 billion on clothes in 2018")
#
# doc = nlp("barack obama is COOL.\nHe is 6 feet\nIs he tall?\nFootball is fun.")
#
# for word in doc:
#     print(word.lemma_)
# # for ent in doc.ents:
# #     print(ent.text, ent.label_)
#
# for sent in doc.sents:
#     print(sent)
# for token in doc:
#     print(token.text)
