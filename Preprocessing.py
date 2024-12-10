import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

model_name = 'microsoft/MiniLM-L12-H384-uncased'
TOKENIZER = AutoTokenizer.from_pretrained(model_name)
MODEL = AutoModel.from_pretrained(model_name)

DF = pd.read_csv('depression.csv')


def process_and_save_embeddings():
    """
    Go through all texts and save the embeddings per the pre trained model
    """
    max_length = 512

    embeddings_dict = {}
    valid_indices = []

    selftext_embeddings = []
    ids = []
    for index, row in DF.iterrows():
        if index % 1000 == 0:
            print(index)
        row_id = row['id']
        selftext_text = row['selftext']

        # Truncated just in case but nothing should be longer than 512
        selftext_tokens_truncated = TOKENIZER(selftext_text, return_tensors="pt", truncation=True, max_length=max_length)

        # Generate embeddings using the model
        with torch.no_grad():
            selftext_embedding = MODEL(**selftext_tokens_truncated).last_hidden_state[:, 0, :]  # [CLS] token embedding

        selftext_embedding = selftext_embedding.squeeze().cpu().numpy()
        embeddings_dict[row_id] = {'selftext': selftext_embedding}

    torch.save(embeddings_dict, 'full_text_embeddings.pt')


def remove_columns():
    """
    Remove the uneeded columns from the dataframe
    """
    df = pd.read_csv("depression-sampled.csv")
    drop_cols = ['created_utc', 'full_link', 'Unnamed: 0']

    filtered_df = df.drop(columns=drop_cols)

    filtered_df.to_csv("depression-data.csv", index=False, encoding='utf-8')


def remove_removed():
    """
    Remove the 'removed' posts from the dataset
    :return:
    """
    new_df = DF[DF["selftext"] != '[removed]']
    embeddings_dict = torch.load('embeddings.pt')
    filtered_ids = set(new_df['id'])

    new_embeddings_dict = {k: v for k, v in embeddings_dict.items() if k in filtered_ids}

    torch.save(new_embeddings_dict, 'embeddings.pt')
    new_df.to_csv('further_filtered_df.csv', index=False, encoding='utf-8')


def remove_deleted_accounts():
    """
    Remove posts from deleted accounts from the dataset
    """
    new_df = DF[DF["author"] != '[deleted]']
    embeddings_dict = torch.load('embeddings.pt')
    filtered_ids = set(new_df['id'])

    new_embeddings_dict = {k: v for k, v in embeddings_dict.items() if k in filtered_ids}

    torch.save(new_embeddings_dict, 'embeddings.pt')
    new_df.to_csv('depression.csv', index=False, encoding='utf-8')


def get_interaction_score():
    """
    Get number of rows that I'll consider "interacted with" (has at least one comment and more than one upvote
    """
    interacted_rows = DF[(DF['num_comments'] > 0) & (DF['score'] > 1)]

    count = len(interacted_rows)
    pass


def add_back_utc():
    """
    Add back UTC column to the dataframe
    """
    df_original = pd.read_csv('depression-sampled.csv')
    DF_filtered = DF.merge(
        df_original[['id', 'created_utc']],  # Select only 'id' and 'utc_created' columns from the original DataFrame
        on='id',  # Merge on the 'id' column
        how='left'  # Keep all rows in DF_filtered, add matching 'utc_created' values
    )
    DF_filtered.to_csv('depression.csv', index=False, encoding='utf-8')


def add_last_post_column():
    """
    Add the 'last_post' column to the dataframe (did this user post again)
    """
    DF['created_utc'] = pd.to_numeric(DF['created_utc'], errors='coerce')

    DF['last_post'] = 0

    max_utc_per_author = DF.groupby('author')['created_utc'].idxmax()

    DF.loc[max_utc_per_author, 'last_post'] = 1

    DF.to_csv('depression.csv', index=False, encoding='utf-8')


def build_posts_dataset():
    """
    Save dataset to a .pt file for loading and training
    """
    embeddings_dict = torch.load('embeddings.pt')

    dataset = []
    for _, row in DF.iterrows():
        post_id = row['id']
        if post_id in embeddings_dict:
            title_embedding = embeddings_dict[post_id]['title']
            softext_embedding = embeddings_dict[post_id]['selftext']
            label = row['last_post']
            dataset.append({
                'title': torch.tensor(title_embedding, dtype=torch.float32),
                'selftext': torch.tensor(softext_embedding, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            })

    torch.save(dataset, 'post_dataset.pt')


def build_second_dataset():
    """
    save the datset to a .pt file for loading and training (second try)
    :return:
    """
    embeddings_dict = torch.load('full_text_embeddings.pt')
    dataset = []
    for _, row in DF.iterrows():
        post_id = row['id']
        if post_id in embeddings_dict:
            softext_embedding = embeddings_dict[post_id]['selftext']
            label = row['last_post']
            dataset.append({
                'selftext': torch.tensor(softext_embedding, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            })

    torch.save(dataset, 'second_post_dataset.pt')


def build_standardized_encodings_dataset():
    """
    Standardize the embeddings to help training
    """
    dataset_path = 'second_post_dataset.pt'
    output_path = 'standardized_encodings_dataset.pt'

    dataset = torch.load(dataset_path)

    all_selftexts = torch.stack([sample['selftext'] for sample in dataset])

    # Compute mean and std across the dataset
    mean = all_selftexts.mean(dim=0)
    std = all_selftexts.std(dim=0)

    # Standardize the dataset
    standardized_dataset = []
    for sample in dataset:
        standardized_selftext = (sample['selftext'] - mean) / (std + 1e-8)
        standardized_sample = {
            'selftext': standardized_selftext,
            'label': sample['label']
        }
        standardized_dataset.append(standardized_sample)

    torch.save(standardized_dataset, output_path)

    torch.save({'mean': mean, 'std': std}, output_path.replace('.pt', '_params.pt'))

    print(f"Standardized dataset saved to {output_path}")
    print(f"Mean and std saved to {output_path.replace('.pt', '_params.pt')}")
