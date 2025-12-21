from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from xmlrpc import client
from lseg.data.content import news, historical_pricing
from datetime import timedelta
import pandas as pd
from typing import List
import refinitiv.data as rd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sqlite3 import Connection
from datetime import datetime
import json
import os
from src.lib.llm_client import LLMClient
from src.logging.logger import logger
from src.workflow.topic import Topic
from src.lib.batch.batch import create_jsonl, create_batches, write_results_from_batches, read_batch_outputs

log = logger(__name__)

pd.options.display.max_colwidth = 100
pd.set_option('future.no_silent_downcasting', True)

def print_start(func):
    def print_start(*args, **kwargs):
        log.info(f"Starting: {func.__name__}")
        return func(*args, **kwargs)
    print_start.__name__ = func.__name__
    print_start.__doc__ = func.__doc__
    return print_start

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
with open(_CONFIG_PATH, "r") as _cfg_f:
    _CFG = json.load(_cfg_f)

_GENERATE_INTERMEDIATE_SUMMARIES = _CFG.get("GENERATE_INTERMEDIATE_SUMMARIES", False)
_MARKET_DATA_PERIOD_HOURS = _CFG.get("MARKET_DATA_PERIOD_HOURS", 1)
_INTERVAL = _CFG.get("INTERVAL", "THIRTY_MINUTES")
if _INTERVAL == "THIRTY_MINUTES":
    _INTERVAL = historical_pricing.Intervals.THIRTY_MINUTES
elif _INTERVAL == "ONE_HOUR":
    _INTERVAL = historical_pricing.Intervals.ONE_HOUR
_COUNT = _CFG.get("SEARCH_COUNT", 1000000)

def fetch_headlines(topic: str, start: str, end: str, count: int) -> pd.DataFrame:
    """Fetch news headlines for a given query and date range."""

    log.debug(f'Fetching headlines for query "{topic}" between {start} and {end}.')

    headlines = news.headlines.Definition(
        query=topic,
        date_from=start,
        date_to=end,
        count=count
    ).get_data().data.df
    
    log.info(f'Found {len(headlines)} stories for query "{topic}" between {start} and {end}.')
    
    return headlines

def fetch_story_bodies(df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
    """Fetch full story bodies for a DataFrame of headlines with storyId column.
    
    Args:
        df: DataFrame with 'storyId' and 'headline' columns, indexed by timestamp
        batch_size: Number of stories to process before logging progress
        
    Returns:
        DataFrame with columns: timestamp, storyId, headline, body
    """
    all_stories = []
    
    story_ids = list(df['storyId'].items())
    total_stories = len(story_ids)
    
    for i in range(0, total_stories, batch_size):
        batch = story_ids[i:i + batch_size]
        batch_data = []
        
        for idx, story_id in batch:
            try:
                response = news.story.Definition(story_id=story_id).get_data()
                if response is None:
                    continue
                body = response.data.raw['newsItem']['contentSet']['inlineData'][0]['$']
                batch_data.append({
                    'timestamp': idx,
                    'storyId': story_id,
                    'headline': df.loc[idx, 'headline'],
                    'body': body
                })
                log.debug(f'Fetched body of storyId {story_id}.')
            except Exception as e:
                log.warning(f'Failed to fetch storyId {story_id}: {e}')
                continue
        
        # Add batch to accumulated stories
        all_stories.extend(batch_data)
        log.info(f'Processed {min(i + batch_size, total_stories)}/{total_stories} stories')
    
    # Convert to DataFrame
    df_stories = pd.DataFrame(all_stories)
    if not df_stories.empty:
        df_stories.set_index('timestamp', inplace=True)
    
    return df_stories

@print_start
def get_stories(start: str, end: str, topic: Topic, refine_search: bool) -> pd.DataFrame:
    topic_news = fetch_headlines(topic.value, start, end, _COUNT)

    topic_news['timestamp'] = topic_news.index

    #top_news = fetch_headlines("TOPNWS", start, end, _COUNT)
    top_news = fetch_headlines("TOPNWS", start, end, _COUNT)

    important_topic = topic_news[['headline', 'storyId', 'timestamp']]#pd.merge(topic_news[['headline', 'storyId', 'timestamp']], top_news[['storyId']], on="storyId")
    important_topic = pd.merge(topic_news[['headline', 'storyId', 'timestamp']], top_news[['storyId']], on="storyId")

    log.info(f'Found {len(important_topic)} important stories for topic {topic.value} between {start} and {end}.')

    if refine_search:
        # Only fetch the latest version of each story
        important_topic['base_storyId'] = important_topic['storyId'].str.rsplit(':', n=1).str[0]
        important_topic['version'] = important_topic['storyId'].str.rsplit(':', n=1).str[1]

        important_topic = important_topic.sort_values('version', ascending=False).groupby('base_storyId').first().reset_index().drop(columns=['version', 'base_storyId'])
        
        # Heuristic: Filter on caps in headline and only grab the stories between 9am and 5pm inclusive
        # important_topic = important_topic[important_topic['headline'].str.isupper()]
        # important_topic = important_topic[
        #     (important_topic['timestamp'].dt.time >= pd.Timestamp('09:00:00').time()) &
        #     (important_topic['timestamp'].dt.time <= pd.Timestamp('17:00:00').time())
        # ]

        log.info(f'After filtering, {len(important_topic)} important stories remain.')

    important_topic.set_index('timestamp', inplace=True)

    # Fetch story bodies using the extracted function
    df_stories = fetch_story_bodies(important_topic)
    
    return df_stories

# Just run this once as it doens't change and burns my request limits. 
@print_start
def get_indices(countries: List[str]) -> pd.DataFrame:
    indices = []
    for country in countries:
        index = rd.discovery.search(
            view = rd.discovery.Views.INDEX_INSTRUMENTS,
            top = 100,
            filter = f"(RCSIndexCountryGroupLeaf eq '{country}' and RCSAssetClass eq 'EQI')",
            select = "RCSAssetClass, CommonName, CommonCode, RIC, IndexCountryGroupName,IndexCountryGroup"
        ).iloc[0]['RIC']

        indices.append(index)

    df_indices = pd.DataFrame({'Country': countries, 'Index': indices})

    return df_indices

def get_market_data_by_index(index: str, timestamp: pd.Timestamp) -> pd.DataFrame:

    log.debug(f"Fetching market data for index {index} at {timestamp}. PID: {os.getpid()}; Thread: {threading.current_thread().name}")
    definition = historical_pricing.summaries.Definition(
        index,
        start=timestamp,
        end=timestamp + timedelta(hours=_MARKET_DATA_PERIOD_HOURS),
        interval = _INTERVAL)
    response = definition.get_data().data.df
    log.debug(f"Market data fetched for index {index} at {timestamp}. PID: {os.getpid()}. Returned {len(response)} rows.")
    _raise_if_rate_limited(response)
    return response

@print_start
def get_market_data(df: pd.DataFrame, output_csv: str, batch_size: int) -> pd.DataFrame:

    df['Data'] = None
    market_data = [None] * len(df)
    
    expanded_rows = []
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(get_market_data_by_index, row['Index'], row['start']): idx 
                        for idx, row in df.iterrows()}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                market_data[idx] = result
                
                # Expand and collect this result
                if isinstance(result, pd.DataFrame) and not result.empty:
                    result['Timestamp'] = result.index
                    data_df = result.copy()
                    data_df['start'] = df.loc[idx, 'start']
                    data_df['Index'] = df.loc[idx, 'Index']
                    # Select only the 4 columns we want
                    data_df = data_df[['Index', 'start', 'Timestamp', 'TRDPRC_1']]
                    expanded_rows.append(data_df)
                    
                processed_count += 1
                
                # Write batch to CSV incrementally
                if len(expanded_rows) >= batch_size:
                    batch_df = pd.concat(expanded_rows, ignore_index=True)
                    # Write headers only if file doesn't exist
                    write_header = not os.path.exists(output_csv)
                    batch_df.to_csv(output_csv, mode='a', header=write_header, index=False)
                    log.info(f"Wrote batch of {len(expanded_rows)} rows to {output_csv} ({processed_count}/{len(df)} processed)")
                    expanded_rows = []
                    
            except ExceededRequestsError:
                log.warning(f"Rate limit exceeded at row {idx}. Saving progress and re-raising.")
                if expanded_rows:
                    batch_df = pd.concat(expanded_rows, ignore_index=True)
                    write_header = not os.path.exists(output_csv)
                    batch_df.to_csv(output_csv, mode='a', header=write_header, index=False)
                raise
            except Exception as e:
                log.warning(f"Error processing index {idx}: {e}")
                market_data[idx] = pd.DataFrame()
    
    # Write any remaining rows
    if expanded_rows:
        batch_df = pd.concat(expanded_rows, ignore_index=True)
        write_header = not os.path.exists(output_csv)
        batch_df.to_csv(output_csv, mode='a', header=write_header, index=False)
        log.info(f"Wrote final batch of {len(expanded_rows)} rows to {output_csv}")
    
    df['Data'] = market_data
    
    return pd.read_csv(output_csv, parse_dates=['start', 'Timestamp'])

@print_start
def run_batch(df: pd.DataFrame, client: LLMClient, file_dir: str) -> pd.DataFrame:
    try:
        create_jsonl(df, f'{file_dir}/input_batch.jsonl', client)
    except Exception as e:
        print(f'Exception occurred while creating JSONL: {e}')
        return None
    try:
        results = create_batches(
            client=client,
            input_batch_file=f'{file_dir}/input_batch.jsonl',
            batch_size=50,
            out_dir=file_dir,
            endpoint='/v1/chat/completions',
            max_concurrent_batches=2
        )
    except Exception as e:
        print(f'Exception occurred while creating batches: {e}')
        return None

    write_results_from_batches(results, f'{file_dir}/output_batch.jsonl', client)

    df_results = read_batch_outputs(outputs_dir=file_dir, glob_pattern='output_batch.jsonl')

    return df_results


@print_start
def generate_ratings(df: pd.DataFrame, client: LLMClient, prompt: str) -> pd.DataFrame:
    ratings = []
    for _, row in df.iterrows():
        try:
            if _GENERATE_INTERMEDIATE_SUMMARIES:
                client.instructions = f"You are an expert financial analyst. Summarize the following news article in a concise manner, maximum 2 sentences:\n\n{row['body']}\n\nSummary:"
                log.debug(f'Generating intermediate summary for storyId {row["storyId"]}.')
                intermediate_response = client.get_response(f"{row['body']}")
                log.debug(f'Intermediate summary for storyId {row["storyId"]}: {intermediate_response}')
                #intermediate_summary = intermediate_response.output_text
                final_input = intermediate_response
            else:
                final_input = row['body']

            client.instructions = prompt
            log.debug(f'Generating final rating for storyId {row["storyId"]}.')
            response = client.get_response(final_input)
            log.debug(f'Final rating for storyId {row["storyId"]}: {response}')
            ratings.append(response)
        except Exception as e:
            print(f"Error generating rating for storyId {row['storyId']}: {e}")
            ratings.append(None)
    df['rating'] = ratings
    #df.drop_duplicates(subset=['timestamp'], inplace=True)
    df = df[df['rating'].notna()]
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').dropna()

    return df

@print_start
def create_regression_data(df_prices: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    
    df_prices['D_TRDPRC_1'] = df_prices.groupby(['Index', 'start'])['TRDPRC_1'].shift(1) - df_prices['TRDPRC_1']
    # instead of shifting, drop where D_TRDPRC_1     is NA
    df_prices = df_prices[df_prices['D_TRDPRC_1'].notna()]
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], errors='coerce')

    data_for_regression = pd.merge(df_prices, ratings, left_on='start', right_on='timestamp', how='inner')
    data_for_regression['Timestamp'] = pd.to_datetime(data_for_regression['Timestamp'], errors='coerce')
    data_for_regression = data_for_regression.drop_duplicates(subset=['Timestamp', 'Index']).dropna()

    return data_for_regression

@print_start
def regress(data_for_regression: pd.DataFrame) -> RegressionResultsWrapper:

    X = pd.to_numeric(data_for_regression['content'], errors='coerce') #.dropna().reset_index(drop=True)
    y = pd.to_numeric(data_for_regression['D_TRDPRC_1'], errors='coerce') #.dropna().reset_index(drop=True)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def _raise_if_rate_limited(response):
    if response is None:
        return
    for attr in ("status_code", "http_status", "status"):
        status = getattr(response, attr, None)
        if status is None:
            continue
        try:
            if int(status) == 429:
                raise ExceededRequestsError("Rate limit exceeded (HTTP 429)")
        except (TypeError, ValueError):
            # non-integer status, ignore and continue checking other attributes
            continue

class ExceededRequestsError(Exception):
    """Raised when an API call returns HTTP 429 Too Many Requests."""
    pass

def check_job_completed(job_id: str, run_id: str, conn: Connection) -> int:
    cur_local = conn.cursor()
    log.debug(f"Checking if job is completed: {job_id}, {run_id}")
    statement = f"SELECT 1 FROM JobCompleted WHERE jobId = '{job_id}' AND runId = '{run_id}' LIMIT 1"
    cur_local.execute(statement)
    if cur_local.fetchone() is not None:
        log.debug(f"Job is completed: {job_id}, {run_id}")
        cur_local.close()
        return 1
    else:
        log.debug(f"Job is not completed: {job_id}, {run_id}")
        cur_local.close()
        return 0

def mark_job_completed(job_id: str, run_id: str, conn: Connection) -> None:
    cur_local = conn.cursor()
    statement = "INSERT INTO JobCompleted (jobId, runId, completed_at) VALUES (?, ?, ?)"
    log.debug(f"Marking job as completed: {job_id}, {run_id}")
    cur_local.execute(
        statement,
        (job_id, run_id, datetime.now())
    )
    conn.commit()
    cur_local.close()
