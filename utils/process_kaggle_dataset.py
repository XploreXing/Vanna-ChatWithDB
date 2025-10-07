import kagglehub
import json
import pandas as pd
import os
output_path ='./data/processed_dataset'
os.makedirs(output_path, exist_ok=True)


def process_kaggle_data():
    """
    Read CSV data from kaggle and process it into three CSV for following population
    """
    first_dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    second_dataset_path = kagglehub.dataset_download("sujaykapadnis/games-on-steam")

    first_file = first_dataset_path + '/games.json'
    with open(first_file, 'r') as file:
        json_data = json.load(file)
    # Remove unnecessary variables
    unnecessary_vars = [
    'packages', 'screenshots', 'movies', 'score_rank', 'header_image',
    'reviews', 'website', 'support_url', 'notes', 'support_email',
    'recommendations', 'user_score', 'median_playtime_forever',
    'median_playtime_2weeks', 'required_age', 'metacritic_score',
    'metacritic_url', 'peak_ccu', 'detailed_description', 'about_the_game',
    'windows', 'mac', 'linux', 'achievements', 'full_audio_languages',
    'genres', 'dlc_count', 'supported_languages', 'developers',
    'publishers', 'average_playtime_forever', 'average_playtime_2weeks',
    'discount'
]
    # Process each game's information and store in a list
    games = [{
    **{k: v for k, v in game_info.items() if k not in unnecessary_vars},
    'tags': list(tags.keys()) if isinstance((tags := game_info.get('tags', {})), dict) else [],
    'tag_frequencies': list(tags.values()) if isinstance(tags, dict) else [],
    'app_id': app_id
} for app_id, game_info in json_data.items()]
    # Create a DataFrame from the processed list
    df = pd.DataFrame(games)

    # Filter games without sales, reviews or categories
    df2 = df[~((df['estimated_owners'] == "0 - 0") | (df['positive'] + df['negative'] == 0) | (df['categories'].str.len() == 0))]
    df2 = df2.copy()
    df2['release_date'] = pd.to_datetime(df2['release_date'], format='mixed')
    df2 = df2[df2['release_date'].dt.year >= 2013]

    # Split estimated_owners into two: min_owners and max_owners
    df2[['min_owners', 'max_owners']] = df2['estimated_owners'].str.split(' - ', expand=True)

    # Remove the original field
    df2 = df2.drop('estimated_owners', axis=1)

    # Remove games with price > $800
    df2 = df2[df2['price'] <= 800]

    second_file = second_dataset_path + '/steamdb.json'
    df_second_dataset = pd.read_json(second_file)
    # Convert 'app_id' integer
    df2['app_id'] = pd.to_numeric(df2['app_id'], errors='coerce').astype('Int64')

    # Perform a left join for 'hltb_single'
    df_merged = pd.merge(df2, df_second_dataset[['sid', 'hltb_single']], left_on='app_id', right_on='sid', how='left')

    # Drop the redundant 'sid' column
    df_merged.drop('sid', axis=1, inplace=True)
    # Limit game duration to 100 hours
    df_merged['hltb_single'] = df_merged['hltb_single'].apply(lambda x: 100 if x > 100 else x)
    df_merged[~(df_merged['hltb_single']>80)]

    # Create a separate DataFrame for each list-type column
    df_categories = df_merged.explode('categories')[['app_id', 'categories']]
    df_tags = df_merged.explode('tags')[['app_id', 'tags']]
    df_frequencies = df_merged.explode('tag_frequencies')['tag_frequencies']
    df_tags['tag_frequencies'] = df_frequencies.values

    # Remove the list columns from the main DataFrame
    columns_to_remove = ['categories', 'tags', 'tag_frequencies']
    df_imploded = df_merged.drop(columns=columns_to_remove)

    # Filter out categories with less than 50 games
    categories_counts = df_categories['categories'].value_counts()
    categories_to_keep = categories_counts[categories_counts >= 50].index.tolist()
    df_categories = df_categories[df_categories['categories'].isin(categories_to_keep)]

    # Filter out tags with less than 50 games
    tags_counts = df_tags['tags'].value_counts()
    tags_to_keep = tags_counts[tags_counts >= 50].index.tolist()
    df_tags = df_tags[df_tags['tags'].isin(tags_to_keep)]

    df_imploded.to_csv(output_path + '/games.csv', index=False)
    df_categories.to_csv(output_path + '/categories.csv', index=False)
    df_tags.to_csv(output_path + '/tags.csv', index=False)

if __name__ == "__main__":
    process_kaggle_data()
    print(f"Processed dataset saved to {output_path}")