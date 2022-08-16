#https://realpython.com/how-to-make-a-discord-bot-python/#:~:text=There%20are%20two%20#key%20steps,and%20implements%20your%#20bot's%20behaviors

# Setting up model ----------------------------------------------------
import string # from some string manipulation tasks
import nltk # natural language toolkit
import re # regex
from string import punctuation # solving punctuation problems
from nltk.corpus import stopwords # stop words in sentences
from nltk.stem import WordNetLemmatizer # For stemming the sentence
from nltk.stem import SnowballStemmer # For stemming the sentence
from contractions import contractions_dict # to solve contractions
from autocorrect import Speller #correcting the spellings
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')



#Libraries for general purpose
import matplotlib.pyplot as plt
import seaborn as sns


#Data preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import pandas as pd

df = pd.read_csv('cyberbullying_tweets.csv')
df['cyberbullying_type'].value_counts()

df.drop(df[df['cyberbullying_type'] == 'other_cyberbullying'].index, inplace = True)
df['cyberbullying_type'].value_counts()

df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
df.sample(10)

df["sentiment"].replace({"religion": 1, "age": 2, "gender": 3, "ethnicity": 4, "not_cyberbullying": 5}, inplace=True)
sentiments = ["religion","age","gender","ethnicity","not bullying"]

import re, string
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



#Remove punctuations, links, stopwords, mentions and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    text =' '.join(word for word in text.split() if len(word) < 14) # remove words longer than 14 characters
    return text

#remove contractions
def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as "&" and "$" present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

#Remove multiple sequential spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)

#Stemming
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

#Lemmatization
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])

#Then we apply all the defined functions in the following order
def preprocess(text):
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    return text
texts_cleaned = []
for t in df.text:
    texts_cleaned.append(preprocess(t))

df['text_clean'] = texts_cleaned
df.head()

df["text_clean"].duplicated().sum()

df.drop_duplicates("text_clean", inplace=True)
df.sentiment.value_counts()

text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)
df['text_len'] = text_len
df.sort_values(by=['text_len'], ascending=False)

df = df[df['text_len'] > 3]
df = df[df['text_len'] < 100]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


tfidf = TfidfTransformer()
clf = CountVectorizer(min_df=1)

X_cv =  clf.fit_transform(df['text_clean'])

tf_transformer = TfidfTransformer(use_idf=True).fit(X_cv)
X_tf = tf_transformer.transform(X_cv)
X_tf

from sklearn.model_selection import train_test_split
# train and test
X_train, X_test, y_train, y_test = train_test_split(X_tf, df['sentiment'], test_size=0.20, stratify=df['sentiment'], random_state=42)
y_train.value_counts()

from imblearn.over_sampling import SMOTE
vc = y_train.value_counts()
while (vc[1] != vc[5]) or (vc[1] !=  vc[3]) or (vc[1] !=  vc[4]) or (vc[1] !=  vc[2]):
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    vc = y_train.value_counts()

y_train.value_counts()

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
# ---------------------------------------------------------------------

# bot.py
from sklearn.linear_model import LogisticRegression
import os
import csv
import discord
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.corpus import stopwords
import difflib
import spacy
from random import randint

load_dotenv()
client = discord.Client()
swear_words = []
with open('bad-words.csv', newline='') as inputfile:
    for row in csv.reader(inputfile):
      swear_words.append(row[0])

conversation_starters = []
with open('conversation-starters.csv', newline='') as inputfile:
    for row in csv.reader(inputfile):
      conversation_starters.append(row[0])
nlp = spacy.load('en_core_web_sm')

convo_subs = []

for x in conversation_starters: 
  doc = nlp(x)
  subject = [tok for tok in doc if (tok.dep_ == "nsubj") ]
  print(type(subject))
  try: 
    temp = subject[0]
    print(temp)
  except: 
    subject = ["aaaaaaaaaaaaaaaa"]
  convo_subs.append(subject)

for x in range(len(convo_subs)): 
  convo_subs[x] = convo_subs[x][0]
  convo_subs[x] = str(convo_subs[x])
  convo_subs[x] = convo_subs[x].lower()
    
print(convo_subs)

global counter_bullying
counter_bullying = 0

def isBullyingHappening(data):
    temp = data

    data = clf.transform(data)
    tf_transformer = TfidfTransformer(use_idf=True).fit(data)
    X_tf_test = tf_transformer.transform(data)
    toPrint = LR.predict(data)
    print(type(toPrint))
    print(toPrint)

    numbullying = 0

    for x in toPrint:
        if x != 5:
            numbullying = numbullying + 1

    if numbullying >= 4:
        return True
    else:
        swears = 0
        for x in temp:
            for i in range(len(swear_words)):
                    if swear_words[i] in x:
                         if "you" in x or "You" in x:
                            swears = swears + 1
                            break
        if numbullying + swears >= 4:
            return True
        else:
            return False

        # replace score with wtw method used to predict sample data
@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
  global counter_bullying
  if counter_bullying >= 24:
      counter_bullying = 0

  counter_bullying = counter_bullying + 1

  data = pd.DataFrame(columns=['content', 'time', 'author'])
  async for msg in message.channel.history():
    if msg.author != client.user:                        
        data = data.append({'content': msg.content,
                                'time': msg.created_at,
                                'author': msg.author.name}, ignore_index=True)
    if len(data['content']) == 10:
      break
  print(data)
  
  text_data = data["content"].values.tolist()
  print("Text data: " + str(text_data))
  final_data = ' '.join(text_data)
  final2 = list(final_data.split(" "))

  print(counter_bullying)
  if isBullyingHappening(text_data) and counter_bullying == 23:
    # 10 messages pass before this is called again
    counter_bullying = 0
    final_text = [word for word in final2 if word not in stopwords.words('english')]
    print(final_text)
  
    for x in final_text: 
      x = x.lower()
    
    counter = Counter(final_text)
    most_common = counter.most_common(1) # Change later
    print(most_common)
  
    final_common = most_common[0]
    ex_final = final_common[0]
    closest = difflib.get_close_matches(ex_final, convo_subs)
    print(closest)
    try: 
      final_topic = closest[0]
      index = convo_subs.index(final_topic)
      await message.channel.send("Hey, that's a sensitive topic. Let's talk about something else. " + conversation_starters[index])
    except: 
      max = len(conversation_starters) - 1
      rand = randint(0, max)
      await message.channel.send("Hey, that's a sensitive topic. Let's talk about something else. " + conversation_starters[rand] + "?")

    # Check if message contains swear words
  text = message.content
  for x in swear_words:
    if x in text:
      if message.author.name == "NoBullying":
        return
      await message.channel.send("Watch your language {}!".format(message.author.mention))
      return

import os
from dotenv import load_dotenv

load_dotenv()
client.run(os.getenv('DISCORD_TOKEN'))