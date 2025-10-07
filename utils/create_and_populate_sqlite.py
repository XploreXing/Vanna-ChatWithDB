import sqlite3
import pandas as pd
import os
data_path ='./data'
os.makedirs(data_path, exist_ok=True)
sqlite_path = data_path + '/db/steam_data.db'

def create_and_populate_sqlite():
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print(f"Removed existing database file {sqlite_path}")
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    
    # Create tables
    init_sqls = """
    CREATE TABLE IF NOT EXISTS games ( 
        name TEXT,
        release_date TEXT,
        price REAL,
        short_description TEXT,
        positive INTEGER,
        negative INTEGER,
        app_id INTEGER PRIMARY KEY,
        min_owners INTEGER,
        max_owners INTEGER,
        hltb_single REAL
    );

    CREATE TABLE IF NOT EXISTS categories (
        app_id INTEGER,
        categories TEXT,
        FOREIGN KEY (app_id) REFERENCES games(app_id)
    );

    CREATE TABLE IF NOT EXISTS tags (
        app_id INTEGER,
        tags TEXT,
        tag_frequencies TEXT,
        FOREIGN KEY (app_id) REFERENCES games(app_id)
    );
    """
    for sql in init_sqls.split(";"):
        c.execute(sql)

    #Read CSV files
    games_csv = data_path + '/processed_dataset/games.csv'
    categories_csv = data_path + '/processed_dataset/categories.csv'
    tags_csv = data_path + '/processed_dataset/tags.csv'

    games_df = pd.read_csv(games_csv)
    categories_df = pd.read_csv(categories_csv)
    tags_df = pd.read_csv(tags_csv)

    #Insert data into tables
    games_df.to_sql('games', conn, if_exists='append', index=False)
    categories_df.to_sql('categories', conn, if_exists='append', index=False)
    tags_df.to_sql('tags', conn, if_exists='append', index=False)

    #Commit and close
    conn.commit()
    conn.close()
    print(f"Database populated successfully: {sqlite_path}")

if __name__ == "__main__":
    create_and_populate_sqlite()