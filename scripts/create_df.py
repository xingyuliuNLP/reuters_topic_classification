### Data extraction

from bs4 import BeautifulSoup
import re
import pandas as pd


def clean_text(text):
    text = text.lower()
    text = re.sub("'s", " 's", text)
    text = re.sub("'ve", " have", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not", text)
    text = re.sub("'re", " are", text)
    text = re.sub("'d", " 'd", text)
    text = re.sub("'ll", " will ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(Reuter|REUTER|reuter)s?(\n&#3;)?', '', text)
    return text

def containTopic(lTest,lFocused):
    '''To filter focused topics'''
    for tp in lTest:
        if tp in lFocused:
            return True
    return False

focused_topics = ['money-fx', 'ship', 'interest', 'acq', 'earn']

## Create empty rows which can be transformed into a dataframe
data = {
            'new_id'     : [],
            'train_test' : [],
            'foc_topics' : [],
            'body'       : [],
            'money-fx'   : [],
            'ship'       : [],
            'interest'   : [],
            'acq'        : [],
            'earn'       : [],
            'multi_topics' : [],
        }

## Load each data file (zfill pads out integers with leading zeros)
content = []
for index in range(22):
    filename = '../raw_data/reut2-{0}.sgm'.format(str(index).zfill(3))
    # test and handle unicode decode errors
    try:
        xml_data = open(filename, 'r', encoding="utf-8").read()
    except UnicodeDecodeError as ude:
        print(f'Failed to read {filename} as utf-8')
        lines = []
        for line in open(filename, 'rb').readlines():
            line = line.decode('utf-8', 'ignore')  # .encode("utf-8")
            lines.append(line)
        xml_data = '\n'.join(lines)
    content.append(xml_data)

## Separate each text file into articles
for i in range(22):
    # Parse text as html using beautiful soup
    parsed_text = BeautifulSoup(content[i], 'html.parser')
    # Focus on NORM texts which have body tag
    for body in parsed_text.find_all('body'):
        id = body.parent.parent['newid'] # extract article id as index
        tt = body.parent.parent['lewissplit'] # extract ModApte Split information for the step of classification
        for topic in body.parent.parent.find_all('topics'):
            d_list = [d.get_text() for d in topic.find_all('d')]
            # multi topics
            if containTopic(d_list, focused_topics):
                data['multi_topics'].append(1) if len(d_list) > 1 else data['multi_topics'].append(0)
            # filtered list
            d_list = list(filter(lambda x: x in focused_topics, d_list))
            # d_list only contains focused topics
            if d_list:
                data['foc_topics'].append(d_list)
                data['body'].append(clean_text(body.get_text()))
                data['new_id'].append(id)
                data['train_test'].append(tt)
                # input 0 and 1s
                for topic in focused_topics:
                    if topic in d_list:
                        data[topic].append(1)
                    else:
                        data[topic].append(0)
df = pd.DataFrame(data)
df.to_csv('reuters.csv', index=False)

