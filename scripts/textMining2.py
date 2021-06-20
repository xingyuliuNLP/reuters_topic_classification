# new era of visualization begins !

import pandas as pd
import re
from matplotlib import pyplot as plt
from collections import Counter
from wordcloud import WordCloud

df = pd.read_csv("../data/reuters_wrangled.csv")

def wordcloud(counter,filename):
    """A small wordloud wrapper"""
    wc = WordCloud(width=1200, height=800,
                   background_color="white",
                   max_words=200)
    wc.generate_from_frequencies(counter)

    # Plot
    fig=plt.figure(figsize=(6, 4),dpi=300)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.show()
    fig.savefig(filename + '.png',bbox_inches='tight',dpi=300)

## overview of the dataframe

df.dtypes

## easy retrieval of articles with different parameter
### here is an example of articles of topic money-fx with or without multilabel

sub_df_multi_topic = df[(df['money-fx']==1) & (df['multi_topics']==1)]
sub_df_single_topic = df[(df['money-fx']==1) & (df['multi_topics']==0)]
print(len(sub_df_single_topic))
print(len(sub_df_multi_topic))

## general statistics
### average score for all the numeric statistics
gen_stat = ["no_tokens","no_types","no_lemmas","avg_wordLength","avg_sentLength","lexical_diversity","lemma_diversity"]
for st in gen_stat:
    print(f"mean of {st}: {df[st].mean()}")

## get diversity score stats and boxplot by topics

focused_topics = ['money-fx', 'ship', 'interest', 'acq', 'earn']

diversity_score = {}
for tp in focused_topics:
    diversity_score[tp] = df[df[tp]==1]['lexical_diversity']
    print(f'average diversity score of {tp}:')
    print(diversity_score[tp].mean())

diversity_score_df = pd.DataFrame(diversity_score)
boxplot_diversity_score = diversity_score_df.boxplot(column=focused_topics)
# boxplot_diversity_score.plot()
plt.savefig('diversity_categories.png',dpi=300)

## get average nb of tokens and nb of types by topics

avg_tokens = {}
for tp in focused_topics:
    avg_tokens[tp] = df[df[tp]==1]['no_tokens']
    print(f'average number of tokens of {tp}:')
    print(avg_tokens[tp].mean())
avg_types = {}
for tp in focused_topics:
    avg_types[tp] = df[df[tp]==1]['no_types']
    print(f'average number of types of {tp}:')
    print(avg_types[tp].mean())

## plot density of word length and sentence length by category
### word length distribution

word_length = {}
for tp in focused_topics:
    word_length[tp] = df[df[tp]==1]['avg_wordLength']
    print(f'average word length of {tp}:')
    print(word_length[tp].mean())

wd_df = pd.DataFrame(word_length)
dens_wd = wd_df.plot.kde()
plt.savefig('word_length_dist.png',bbox_inches='tight',dpi=300)

### sentence length distribution
sent_length = {}
for tp in focused_topics:
    sent_length[tp] = df[df[tp]==1]['avg_sentLength']
    print(f'average sentence length of {tp}:')
    print(sent_length[tp].mean())
st_df = pd.DataFrame(sent_length)
dens_sent = st_df.plot.kde()
plt.savefig('sentence_length_dist.png',bbox_inches='tight',dpi=300)


# loop to create all the plots for most common words by topic
counter_dict = {}
for tp in focused_topics:
    lemmas_focused = []
    i_df = df[df[tp] == 1]
    for i, row in i_df.iterrows():
        lemmas_focused.append(row['body_lemma_noStop'])
    ### split all the sentences into a list of words
    lemmas_focused_count = []
    for wl in lemmas_focused:
        lemmas_focused_count += wl.split(' ')
    counter = Counter(lemmas_focused_count)
    freq_df = pd.DataFrame.from_records(counter.most_common(10),
                                        columns=['words', 'count']).sort_values('count',ascending=False)
    # create bar plot
    freq_df.plot(kind='barh', x='words')
    plt.savefig('mostCommon' + tp + '.png', bbox_inches='tight', dpi=300)
    print(f'10 most common words of {tp} are: ')
    print(counter.most_common(10))
    counter_dict[tp] = counter

## word cloud for frequent words of each topic

for k,v in counter_dict.items():
    wordcloud(v,'cloud'+k)

## loop to create all the plots for most common people names by topic
counter_dict = {}
for tp in focused_topics:
    lemmas_focused = []
    i_df = df[df[tp] == 1]
    for i, row in i_df.iterrows():
        lemmas_focused.append(row['body_people'])
    ### split all the sentences into a list of words
    lemmas_focused_count = []
    for wl in lemmas_focused:
        if type(wl) != float:
            lemmas_focused_count += wl.split(',')
    counter = Counter(lemmas_focused_count)
    freq_df = pd.DataFrame.from_records(counter.most_common(10),
                                        columns=['people', 'count'])
    # create bar plot
    freq_df.plot(kind='barh', x='people')
    plt.savefig('mostCommonNames' + tp + '.png', bbox_inches='tight', dpi=300)
    print(f'10 most common names of {tp} are: ')
    print(counter.most_common(10))
    counter_dict[tp] = counter

## word cloud for people names by topic

for k,v in counter_dict.items():
    wordcloud(v,'cloudPeopleNames'+k)

## loop to create all the plots for most common org names by topic
counter_dict = {}
for tp in focused_topics:
    lemmas_focused = []
    i_df = df[df[tp] == 1]
    for i, row in i_df.iterrows():
        lemmas_focused.append(row['body_org'])
    ### split all the sentences into a list of words
    lemmas_focused_count = []
    for wl in lemmas_focused:
        if type(wl) != float:
            lemmas_focused_count += wl.split(',')
    counter = Counter(lemmas_focused_count)
    freq_df = pd.DataFrame.from_records(counter.most_common(10),
                                        columns=['orgs', 'count'])
    # create bar plot
    freq_df.plot(kind='barh', x='orgs')
    plt.savefig('mostCommonOrgs' + tp + '.png', bbox_inches='tight', dpi=300)
    print(f'10 most common orgs of {tp} are: ')
    print(counter.most_common(10))
    counter_dict[tp] = counter

## wordclouds for org names by topic
for k,v in counter_dict.items():
    wordcloud(v,'cloudOrgNames'+k)

## loop to create all the plots for most common quantity names by topic

counter_dict = {}
for tp in focused_topics:
    lemmas_focused = []
    i_df = df[df[tp] == 1]
    for i, row in i_df.iterrows():
        lemmas_focused.append(row['body_quantity'])
    ### split all the sentences into a list of words
    lemmas_focused_count = []
    for wl in lemmas_focused:
        if type(wl) != float:
            lemmas_focused_count += wl.split(',')
    str = ','.join(lemmas_focused_count)
    ## remove floats
    str = re.sub(r'\d+\.?\d+','',str)
    ## remove single digit
    str = re.sub(r'\d','',str)
    ## remove symbols
    str = re.sub(r'(\-|\/)','',str)
    ## remove more than spaces
    str = re.sub(r' {2,}','',str)
    # replace numbers and spaces
    l_tokens = str.split(',')
    counter = Counter(l_tokens)
    # clean the empty string entry
    del counter['']
    del counter[' ']
    freq_df = pd.DataFrame.from_records(counter.most_common(10),
                                        columns=['quantity', 'count'])
    # create bar plot
    freq_df.plot(kind='barh', x='quantity')
    plt.savefig('mostCommonQuantities' + tp + '.png', bbox_inches='tight', dpi=300)
    print(f'10 most common quantities of {tp} are: ')
    print(counter.most_common(10))
    counter_dict[tp] = counter

## word clouds for quantity names by topic
for k,v in counter_dict.items():
    wordcloud(v,'cloudQuantityNames'+k)