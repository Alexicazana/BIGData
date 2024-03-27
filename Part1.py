from datasets import load_dataset
import pandas as pd
# import spacy
# spacy.load("en_core_web_sm")

# import en_core_web_sm
# nlp = en_core_web_sm.load()

import nltk
nltk.download("stopwords")

from gensim.parsing.preprocessing import remove_stopwords

import numpy as np


import dask.dataframe as dd
import dask.bag as db
from dask.diagnostics import ProgressBar




#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


#visualize
# import pyLDAvis
# import pyLDAvis.gensim

###Parallel Processing
from joblib import Parallel, delayed

df_openstax = dd.read_parquet("Openstax/*.parquet")
df_khan = dd.read_parquet("Khan/*.parquet")
df_stanford = dd.read_parquet("Stanford/*.parquet")

text1 = df_openstax['text'].compute().tolist()
text2 = df_khan['text'].compute().tolist()
text3 = df_stanford['text'].compute().tolist()

# df_all = dd.concat([df_openstax, df_khan, df_stanford])
# df_repartitioned = df_all.repartition(npartitions=40)

sum_text = text1 + text2 + text3

# print(sum_text[:5])

# Convert the list to a DataFrame
texts_df = pd.DataFrame(sum_text, columns=['text'])

# Save the DataFrame as a Parquet file
texts_df.to_parquet('lda_texts.parquet')

texts_ddf = dd.read_parquet('lda_texts.parquet')
# print(texts_df.head())

texts = db.from_sequence(texts_ddf, npartitions=16)



# def remove_stopwords(texts):
#     results = []
#     for text in texts:
#         new_text = remove_stopwords(text)
#         results.append(new_text)
#     return results

# def lemmatize_batch(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
#     results = []
#     for text in texts:
#         doc = nlp(text)
#         new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
#         results.append(" ".join(new_text))
#     return results
def lemmatize_batch(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
  def preprocess_partition(text_partition):
      results = []
      for text in text_partition:
          preprocessed_text = preprocess_text(text, allowed_postags=allowed_postags)
          results.append(preprocessed_text)
      return results
  return texts.map_partitions(preprocess_partition)

import dask.bag as db

def preprocess_text(text, stopwords=stopwords.words('english'), allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
  # ... (your preprocessing logic here) ...
  return preprocessed_text

texts_bag = db.from_sequence(texts_ddf)
lemmatized_texts_bag = texts_bag.map(preprocess_text)

# compute the results when needed 
with ProgressBar():
    lemmatized_texts = lemmatized_texts_bag.compute()


# Process texts in parallel using Dask Bag
# lemmatized_texts_bag = texts_ddf.map_partitions(lemmatize_batch, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"])

# # Compute the result (this is where the actual computation happens)
# with ProgressBar():
#     lemmatized_texts = lemmatized_texts_bag.compute()




# def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
#     nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#     texts_out = []
#     for text in texts:
#         doc = nlp(text)
#         new_text = []
#         for token in doc:
#             if token.pos_ in allowed_postags:
#                 new_text.append(token.lemma_)
#         final = " ".join(new_text)
#         texts_out.append(final)
#     return texts_out

# # Load the Parquet file into a DataFrame
# texts_df = pd.read_parquet('lda_texts.parquet')

# # Ensure you are passing the correct column to the function
# lemmatized_texts = lemmatization(texts_df['text'][:100])
# print(lemmatized_texts[:5])



# def lemmatize_document(doc,allowed_postags=["NOUN","ADJ","VERB","ADV"]):
#   new_text=[]
#   for token in doc:
#     if token.pos_ in allowed_postags:
#         new_text.append(token.lemma_)
#     final = " ".join(new_text)
#     return final

# # def lemmatization(texts, n_jobs=-1, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
# #     """Parallel lemmatization of texts."""
# #     # Processing texts in parallel
# #     lemmatized_texts = Parallel(n_jobs=n_jobs)(
# #         delayed(lemmatize_document)(nlp(text), allowed_postags) for text in texts
# #     )
# #     return lemmatized_texts
 

# texts_df = pd.read_parquet('lda_texts.parquet')
# lemmatized_texts = lemmatization(texts_df['text'][:1000].tolist())

# print(lemmatized_texts[:5])



# def gen_words(texts):
#   final = []
#   for text in texts:
#     new = gensim.utils.simple_preprocess(text,deacc=True)
#     final.append(new)
#   return final

# data_words=gen_words(lemmatized_texts)

# print(data_words[:5])



# id2word = corpora.Dictionary(data_words)
# id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

# corpus = []
# for text in data_words:
#   new = id2word.doc2bow(text)
#   corpus.append(new)

# print(corpus[:50])


# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=15,random_state=100,update_every=1,chunksize=100,passes=10,alpha="auto")

