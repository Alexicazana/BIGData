from datasets import load_dataset
import pandas as pd
import dask.dataframe as dd
import spacy
import nltk
nltk.download("stopwords")
import numpy as np
import json
import glob

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


#spacy
from nltk.corpus import stopwords

#visualize
# import pyLDAvis
# import pyLDAvis.gensim

###Parallel Processing
from joblib import Parallel, delayed


khan = load_dataset("HuggingFaceTB/cosmopedia", "khanacademy",split="train")
openstax = load_dataset("HuggingFaceTB/cosmopedia", "openstax", split="train")
stanford = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train")

partitions_stanford = [delayed(pd.DataFrame)(d) for d in stanford]
stanford_dask_dataframe = dd.from_delayed(partitions_stanford)

partitions_khan = [delayed(pd.DataFrame)(d) for d in khan]
khan_dask_dataframe = dd.from_delayed(partitions_khan)

partitions_openstax = [delayed(pd.DataFrame)(d) for d in openstax]
openstax_dask_dataframe = dd.from_delayed(partitions_openstax)



text1 = filtered_stanford_df['text'].tolist()
text2 = filtered_openstax_df['text'].tolist()
text3 = filtered_khan_df['text'].tolist()

all_texts = text1 + text2 + text3

# Convert the list to a DataFrame
texts_df = pd.DataFrame(all_texts, columns=['text'])

# Save the DataFrame as a Parquet file
texts_df.to_parquet('lda_texts.parquet')

texts_df = pd.read_parquet('lda_texts.parquet')
print(texts_df.head())

stopwords = stopwords.words("english")
# print(stopwords)
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

nlp = spacy.load("en_core_web_sm",disable=["parser","ner"])


#### Custom stop words
# custom_stopwords = ['specificTerm1', 'specificTerm2']  # Add your domain-specific terms here
# Combine the stop word lists, ensuring uniqueness
# extended_stopwords = set(nltk_stopwords + list(spacy_stopwords) + custom_stopwords)
#######






def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out

# Load the Parquet file into a DataFrame
texts_df = pd.read_parquet('lda_texts.parquet')

# Ensure you are passing the correct column to the function
lemmatized_texts = lemmatization(texts_df['text'][:100])
print(lemmatized_texts[:5])



# def lemmatize_document(doc,allowed_postags=["NOUN","ADJ","VERB","ADV"]):
#   new_text=[]
#   for token in doc:
#     if token.pos_ in allowed_postags:
#         new_text.append(token.lemma_)
#     final = " ".join(new_text)
#     return final

# def lemmatization(texts, n_jobs=-1, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
#     """Parallel lemmatization of texts."""
#     # Processing texts in parallel
#     lemmatized_texts = Parallel(n_jobs=n_jobs)(
#         delayed(lemmatize_document)(nlp(text), allowed_postags) for text in texts
#     )
#     return lemmatized_texts
 

texts_df = pd.read_parquet('lda_texts.parquet')
lemmatized_texts = lemmatization(texts_df['text'][:1000].tolist())

print(lemmatized_texts[:5])



def gen_words(texts):
  final = []
  for text in texts:
    new = gensim.utils.simple_preprocess(text,deacc=True)
    final.append(new)
  return final

data_words=gen_words(lemmatized_texts)

print(data_words[:5])



id2word = corpora.Dictionary(data_words)
id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

corpus = []
for text in data_words:
  new = id2word.doc2bow(text)
  corpus.append(new)

print(corpus[:50])


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=15,random_state=100,update_every=1,chunksize=100,passes=10,alpha="auto")

