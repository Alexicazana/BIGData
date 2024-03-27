import pandas as pd
from dask.distributed import Client
import nltk
import dask.dataframe as dd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import multiprocessing as mp
from pandas import NA



# Function to preprocess text
def preprocess_text(text):
    # Handle missing values: return an empty string if text is NA or None
    if text is NA or text is None:
        return ''
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize and remove stopwords
    words = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]
    
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back to string
    return ' '.join(lemmatized_words)

def main():
    client = Client()

    # Download necessary NLTK data
    nltk.download("stopwords")
    nltk.download('punkt')
    nltk.download('wordnet')

    # Read the parquet files into Dask DataFrames
    df_openstax = dd.read_parquet("Openstax/*.parquet")
    df_khan = dd.read_parquet("Khan/*.parquet")
    df_stanford = dd.read_parquet("Stanford/*.parquet")

    # Concatenate all DataFrames
    df_all = dd.concat([df_openstax, df_khan, df_stanford])

    # Apply preprocessing to the 'text' column
    df_all['processed_text'] = df_all['text'].map_partitions(lambda df: df.apply(preprocess_text))

    # Convert the processed Dask DataFrame to pandas DataFrame for further use (if necessary)
    processed_texts_df = df_all.compute()

    # Optionally, save the processed texts back to a Parquet file
    processed_texts_df.to_parquet('processed_lda_texts.parquet')

if __name__ == '__main__':
    mp.set_start_method('fork')  # or 'spawn' to test different behaviors
    main()
