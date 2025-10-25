import uuid
import argparse
import lseg.data as ld
from openai import OpenAI
import refinitiv.data as rd
import pandas as pd
from functions import get_data, get_indices, get_market_data, generate_ratings, create_regression_data, regress, ExceededRequestsError, mark_job_completed, check_job_completed 
from getpass import getpass
from enum import IntEnum
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
OPEN_AI_SECRET_KEY = os.getenv("OPEN_AI_SECRET_KEY")
NEBIUS_SECRET_KEY = os.getenv("NEBIUS_SECRET_KEY")

class Stage(IntEnum):
    get_data = 0
    get_market_data = 1
    generate_ratings = 2
    regress = 3

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MacroNews pipeline")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--stage", required=True, choices=["get_data", "get_market_data", "generate_ratings", "regress"], help="Stage to run: get_data/get_market_data/generate_ratings/regress")
    return p.parse_args()


_CLIENT = OpenAI(
    # api_key=getpass("Enter your OpenAI API key: ")
    api_key=OPEN_AI_SECRET_KEY
)

_DATABASE = 'data/database/llms_macronews.db'

_SEARCH_COUNT = 1000000

def run(start: str, end: str, stage: Stage, conn: sqlite3.Connection):

    run_id = str(uuid.uuid4())
    print(f"Running pipeline from stage={stage} from {start} to {end}")
    print(f"RunId: {run_id} assigned.")

    cur = conn.cursor()
    cur.execute("INSERT INTO Run (run_id, start_date, end_date, model_id, prompt_id, topic_id) VALUES (?, ?, ?, 1, 1, 1)", (run_id, start, end))
    conn.commit()
    cur.close()

    file_dir = f'data/{run_id}'
    os.makedirs(file_dir, exist_ok=True) 

    if stage <= Stage.get_data:
        stories = get_data(run_id, _SEARCH_COUNT, conn)
        stories.to_csv(f'{file_dir}/stories.csv', index=False)
    else:
        stories = pd.read_csv(f'{file_dir}/stories.csv', parse_dates=['timestamp'])

    indices = pd.read_csv(f'data/indices.csv')

    if stage <= Stage.get_market_data:
        df_prices = pd.merge(
            indices.assign(key=1),
            pd.DataFrame({'start': pd.to_datetime(stories['timestamp'], format='ISO8601')}).assign(key=1),
            on='key'
        ).drop('key', axis=1)

        try:
            df_prices = get_market_data(run_id, df_prices, conn)              
        except ExceededRequestsError:
            print(f"Error fetching market data: Rate limit exceeded.")
            raise
        finally:
            # Whether get_market_data has succeeded or failed, save whatever data we have so far
            expanded_rows = []
            for _, row in df_prices.iterrows():
                if isinstance(row['Data'], pd.DataFrame): 
                    row['Data']['Timestamp'] = row['Data'].index
                    data_df = row['Data'].copy()
                    data_df['start'] = row['start']
                    data_df['Index'] = row['Index']
                    expanded_rows.append(data_df)
            df_prices = pd.concat(expanded_rows, ignore_index=True)
            df_prices.to_csv(f'{file_dir}/df_prices.csv', index=False)

    else:
        df_prices = pd.read_csv(f'{file_dir}/df_prices.csv', parse_dates=['start', 'Timestamp'])[['Index', 'start', 'Timestamp', 'TRDPRC_1']]

    if stage <= Stage.generate_ratings:
        with open(f'data/prompt.txt') as f:
            prompt = f.read()
        ratings = generate_ratings(run_id, stories, _CLIENT, prompt, conn)
        ratings.to_csv(f'{file_dir}/ratings.csv', index=False)
    else:
        ratings = pd.read_csv(f'{file_dir}/ratings.csv', parse_dates=['timestamp'])
        ratings.dropna(subset=['rating'], inplace=True)

    if stage <= Stage.regress:
        regression_data = create_regression_data(df_prices, ratings)
        model = regress(regression_data)
        print(model.summary())


if __name__ == "__main__":
    args = parse_args()
    stage = Stage[args.stage]
    ld.open_session()
    rd.open_session()
    conn = sqlite3.connect(_DATABASE)
    try:
        run(args.start, args.end, stage, conn)
        conn.commit()
    except Exception as e:
        print(f"Error occurred: {e}")
        conn.rollback()
    finally:
        ld.close_session()
        rd.close_session()
        conn.close()

