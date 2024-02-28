# Exploration 
import pandas as pd
from datasets import load_dataset
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
import torch
import csv

stanford_dataset = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train")
df = pd.DataFrame(stanford_dataset, columns=stanford_dataset.column_names)
min_token_length = df['text_token_length'].min()
max_token_length = df['text_token_length'].max()
mean_token_length = df['text_token_length'].mean()
median_token_length = df['text_token_length'].median()

# khan_dataset = load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train")
# openstax_dataset = load_dataset("HuggingFaceTB/cosmopedia", "openstax", split="train")

with open('stanford.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["min_token_length", "max_token_length", "mean_token_length", "median_token_length"])
    writer.writerow([min_token_length, max_token_length, mean_token_length, median_token_length])
    for i in range(5):
        writer.writerow([df['text_token_length']])
    
    # writer = csv.writer(file)
    # writer.writerow(["text_token_length"])
    # for i in range(5):
    #     writer.writerow([df(stanford_dataset)['text_token_length']])

# print(stanford_dataset[0])
# df['token_l'].mean()
# df['col_name'].std()
# df['col_name'].max()
# df['col_name'].min()
# df['col_name'].median()
# df['col_name'].count()
# df['col_name'].describe()
# df['col_name'].unique()
# df['col_name'].nunique()
