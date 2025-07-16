import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_dir = 'data'
fake_csv_path = os.path.join(data_dir, 'Fake.csv')
true_csv_path = os.path.join(data_dir, 'True.csv')

if not os.path.exists(fake_csv_path) or not os.path.exists(true_csv_path):
    print(f"Error: Make sure 'Fake.csv' and 'True.csv' are in the '{data_dir}' directory.")
else:
    df_fake = pd.read_csv(fake_csv_path)
    df_true = pd.read_csv(true_csv_path)

    df_fake['label'] = 0 # 0 for Fake news
    df_true['label'] = 1 # 1 for True news

    # Combine the dataframes
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Shuffle the dataset to mix the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("--- Dataset Information ---")
    df.info()

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Combine title and text for a complete news feature
    df['full_text'] = df['title'] + " " + df['text']

    print("\n--- First 5 Rows of Combined Data ---")
    print(df.head())

    # Visualizations
    print("\nGenerating Class Distribution plot...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution (0: Fake, 1: Real)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.show()

    print("Generating Text Length Distribution plot...")
    df['text_length'] = df['full_text'].str.len()
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
    plt.title('Distribution of News Article Length')
    plt.xlabel('Length of Text')
    plt.ylabel('Frequency')
    plt.legend(title='Label', labels=['Real', 'Fake'])
    plt.show()

    print("\nExploratory Data Analysis complete!")