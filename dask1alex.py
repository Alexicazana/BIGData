import dask.bag as db
from dask.diagnostics import ProgressBar
import spacy

# Create a Dask Bag from the list of texts. Each partition is a list of texts.
texts_bag = db.from_sequence(lemmatized_texts, npartitions=10)

def lemmatize_batch(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    results = []
    for text in texts:
        doc = nlp(text)
        new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        results.append(" ".join(new_text))
    return results

# Process texts in parallel using Dask Bag
lemmatized_texts_bag = texts_bag.map_partitions(lemmatize_batch, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"])

# Compute the result (this is where the actual computation happens)
with ProgressBar():
    lemmatized_texts = lemmatized_texts_bag.compute()
