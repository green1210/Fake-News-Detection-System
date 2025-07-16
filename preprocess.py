import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 256
OOV_TOKEN = "<OOV>" 

print("Loading and combining data...")
df_fake = pd.read_csv('data/Fake.csv')
df_true = pd.read_csv('data/True.csv')

df_fake['label'] = 0
df_true['label'] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['full_text'] = df['title'] + " " + df['text']
print("Data loading complete.")

print("Tokenizing text...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(df['full_text'])

sequences = tokenizer.texts_to_sequences(df['full_text'])


padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
print("Tokenization complete.")

print("Saving tokenizer to 'tokenizer.pickle'...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved.")


print("Splitting and saving data...")
X = padded_sequences
y = df['label'].values

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/X_test.npy', X_test)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/y_test.npy', y_test)
print("Processed data saved in 'processed_data/' directory.")

print("\n--- Preprocessing Summary ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")
print("\nPreprocessing complete!")