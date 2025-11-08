import traceback
from tracemalloc import start
import uuid
import argparse
import lseg.data as ld
from openai import OpenAI
import refinitiv.data as rd
import pandas as pd
from src.lib.functions import get_stories, get_indices, get_market_data, generate_ratings, create_regression_data, regress, check_job_completed, mark_job_completed
import sqlite3
import os
from dotenv import load_dotenv
import json
import traceback
from src.logging.logger import logger
from src.lib.llm_client import LLMClient
from src.workflow.stage import Stage
from src.workflow.topic import Topic
load_dotenv()
OPEN_AI_SECRET_KEY = os.getenv("OPEN_AI_SECRET_KEY")
NEBIUS_SECRET_KEY = os.getenv("NEBIUS_SECRET_KEY")

log = logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline")
    p.add_argument("--start", required=False, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=False, help="End date (YYYY-MM-DD)")
    p.add_argument("--run_id", required=False, help="ID of an existing run")
    p.add_argument("--stage", required=False, choices=["get_news", "get_market_data", "generate_ratings", "regress"], help="Stage to run: get_news/get_market_data/generate_ratings/regress")
    args = p.parse_args()

    if args.run_id:
        if args.start or args.end:
            p.error("Provide either --run_id OR both --start and --end, not both.")
    else:
        if not (args.start and args.end):
            p.error("You must provide either --run_id OR both --start and --end.")

    return args

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
with open(_CONFIG_PATH, "r") as _cfg_f:
    _CFG = json.load(_cfg_f)

_SEARCH_COUNT = _CFG.get("SEARCH_COUNT", 1000000)
_DATABASE = _CFG.get("DATABASE", "../../data/database/llms_macronews.db")
_BATCH_SIZE = _CFG.get("MARKET_DATA_BATCH_SIZE", 10000)
_INDICES_LIST = _CFG.get("COUNTRIES", "indices_majors.csv")
_INDICES = pd.read_csv(f'data/{_INDICES_LIST}')
_MARKET_DATA_PERIOD_HOURS = _CFG.get("MARKET_DATA_PERIOD_HOURS", 1)
_TOPIC = _CFG.get("TOPIC", "U.S. Federal Reserve")

def run(run_id: str, stage: Stage, conn: sqlite3.Connection):

    log.info(f"Running pipeline from stage={stage.value}")
    
    file_dir = f'data/runs/{run_id}'
    os.makedirs(file_dir, exist_ok=True)
    
    with open(f'data/prompt.txt') as f:
        prompt = f.read()

    client = LLMClient(model=_CFG.get("MODEL", "gpt-4o"), instructions=prompt)

    cur = conn.cursor()
    cur.execute("SELECT start_date, end_date FROM Run WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No run found with run_id: {run_id}")
    start_date, end_date = row
    cur.close()

    topic = Topic[_TOPIC]

    if stage <= Stage.get_news:
        if not check_job_completed("get_news", run_id, conn):
            ld.open_session()
            rd.open_session()
            stories = get_stories(start_date, end_date, _SEARCH_COUNT, topic)
            stories.to_csv(f'{file_dir}/stories.csv', index=True)
            mark_job_completed("get_news", run_id, conn)
            ld.close_session()
            rd.close_session()
        else:
            log.info("get_news job already completed, loading from CSV.")
            stories = pd.read_csv(f'{file_dir}/stories.csv', index_col=0, parse_dates=[0])
    else:
        stories_csv = f'{file_dir}/stories.csv'
        log.info(f'{Stage.get_news.name} stage skipped. Loading stories from {stories_csv}.')        
        stories = pd.read_csv(stories_csv, index_col=0, parse_dates=[0])  # parse index as datetime

    indices = _INDICES

    if stage <= Stage.get_market_data:
        if not check_job_completed("get_market_data", run_id, conn):
            df_prices = pd.merge(
                indices.assign(key=1),
                pd.DataFrame({'start': stories.index}).assign(key=1),
                on='key'
            ).drop('key', axis=1)

            ld.open_session()
            rd.open_session()

            output_csv = f'{file_dir}/df_prices.csv'

            try:
                df_prices = get_market_data(df_prices, output_csv, _BATCH_SIZE)
                mark_job_completed("get_market_data", run_id, conn)
            except Exception as e:
                print(f"Error fetching market data: {e}. Partial results may be saved to {output_csv}")
                raise
            finally:
                ld.close_session()
                rd.close_session()
        else:
            log.info("get_market_data job already completed, loading from CSV.")
            df_prices = pd.read_csv(f'{file_dir}/df_prices.csv', header=0, parse_dates=['start', 'Timestamp'])
    else:
        prices_csv = f'{file_dir}/df_prices.csv'
        log.info(f'{Stage.get_market_data.name} stage skipped. Loading market data from {prices_csv}.')
        df_prices = pd.read_csv(prices_csv, header=0, parse_dates=['start', 'Timestamp'])

    if stage <= Stage.generate_ratings:
        if not check_job_completed("generate_ratings", run_id, conn):
            stories.drop_duplicates(inplace=True)
            ratings = generate_ratings(stories, client, prompt)
            ratings.to_csv(f'{file_dir}/ratings.csv', index=True)
            mark_job_completed("generate_ratings", run_id, conn)
        else:
            log.info("generate_ratings job already completed, loading from CSV.")
            ratings = pd.read_csv(f'{file_dir}/ratings.csv', header=0, parse_dates=['timestamp'])
            ratings.dropna(subset=['rating'], inplace=True)
    else:
        ratings_csv = f'{file_dir}/ratings.csv'
        log.info(f'{Stage.generate_ratings.name} stage skipped. Loading ratings from {ratings_csv}.')
        ratings = pd.read_csv(ratings_csv, header=0, parse_dates=['timestamp'])
        ratings.dropna(subset=['rating'], inplace=True)

    if stage <= Stage.regress:  
        regression_data = create_regression_data(df_prices, ratings)
        results = regress(regression_data)
        save_results(results, run_id)
    


def save_results(results: pd.DataFrame, run_id: str) -> None:
    rating_coef = results.params['rating'] if 'rating' in results.params.index else results.params.iloc[1]
    rating_pvalue = results.pvalues['rating'] if 'rating' in results.pvalues.index else results.pvalues.iloc[1]

    result_row = {
        'run_id': run_id,
        'topic': _TOPIC,
        'model': _CFG.get("MODEL", ""),
        'indices': ','.join(_INDICES['Index'].tolist()),
        'market_data_period_hours': _MARKET_DATA_PERIOD_HOURS,
        'rating_coefficient': rating_coef,
        'rating_pvalue': rating_pvalue,
    }
    
    result_df = pd.DataFrame([result_row])
    output_file = f'data/runs/{run_id}/results.csv'
    result_df.to_csv(output_file, mode='a', header=True, index=False)

    log.info(f'Saved results to {output_file}')

    return None
    

if __name__ == "__main__":
    args = parse_args()
    stage = Stage[args.stage]
    conn = sqlite3.connect(_DATABASE)

    try:
        if not args.run_id: 
            run_id = str(uuid.uuid4())
            print(f"RunId: {run_id} assigned.")
            cur = conn.cursor()
            cur.execute("INSERT INTO Run (run_id, start_date, end_date, model, prompt, topic) VALUES (?, ?, ?, 1, 1, 1)", (run_id, args.start, args.end))
            conn.commit()
            cur.close()
            run(run_id, stage, conn)    
        else:    
            run(args.run_id, stage, conn)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.commit()
        conn.close()
