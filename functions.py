from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import lseg.data as ld
from lseg.data.content import news, historical_pricing
from datetime import timedelta
import pandas as pd
from typing import List
import refinitiv.data as rd
from openai import OpenAI
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sqlite3 import Connection
from datetime import datetime
pd.options.display.max_colwidth = 100
pd.set_option('future.no_silent_downcasting', True)

def print_start(func):
    def wrapper(*args, **kwargs):
        print(f"Starting: {func.__name__}")
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

_INTERVAL = historical_pricing.Intervals.THIRTY_MINUTES
_MARKET_DATA_PERIOD_HOURS = 1
_MODEL = "gpt-4o"
_GENERATE_INTERMEDIATE_SUMMARIES = 1

@print_start
def get_data(run_id: str, count: int, conn: sqlite3.Connection) -> pd.DataFrame:

    cur = conn.cursor()
    cur.execute("SELECT start_date, end_date FROM Run WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No run found with run_id: {run_id}")
    start_date, end_date = row
    cur.close()

    response = news.headlines.Definition(
        query="Fed",
        date_from=start_date,
        date_to=end_date, 
        count=count
    ).get_data()
    fed = response.data.df

    topnews = news.headlines.Definition(
        query="TOPNWS",
        date_from=start_date, 
        date_to=end_date, 
        count=count
    ).get_data().data

    important_fd = pd.merge(fed, topnews.df, on="storyId")
    important_fd = important_fd[important_fd['headline_x'].str.isupper()]

    timestamps = []
    bodies = []
    valid_idx = []

    for idx, story_id in important_fd['storyId'].items():
        response = news.story.Definition(story_id=story_id).get_data()
        if response is None:
            continue
        try:
            ts = response.data.raw['newsItem']['itemMeta']['versionCreated']['$']
            body = response.data.raw['newsItem']['contentSet']['inlineData'][0]['$']
        except Exception:
            continue

        timestamps.append(ts)
        bodies.append(body)
        valid_idx.append(idx)

    important_fd = important_fd.loc[valid_idx].reset_index(drop=True)
    important_fd['timestamp'] = pd.to_datetime(timestamps)
    important_fd['body'] = bodies
    df_stories = important_fd[['headline_x', 'storyId', 'timestamp', 'body']]

    # Only grab the stories between 9am and 5pm
    df_stories = df_stories[
        (df_stories['timestamp'].dt.time >= pd.Timestamp('09:00:00').time()) &
        (df_stories['timestamp'].dt.time <= pd.Timestamp('17:00:00').time())
    ]

    mark_job_completed("get_data", run_id, conn)
    
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

    definition = historical_pricing.summaries.Definition(
        index,
        interval=_INTERVAL,
        start=timestamp,
        end=timestamp + timedelta(hours=_MARKET_DATA_PERIOD_HOURS))
    response = definition.get_data().data.df
    _raise_if_rate_limited(response)
    return response

@print_start
def get_market_data(run_id: str, df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:

    if check_job_completed("get_market_data", run_id, conn):
        raise Exception(f"Job get_market_data already completed for runId: {run_id}")

    df['Data'] = None
    market_data = [None] * len(df)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(get_market_data_by_index, row['Index'], row['start']): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                market_data[idx] = future.result()
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                market_data[idx] = None
    df['Data'] = market_data

    mark_job_completed("get_market_data", run_id, conn)
    return df

@print_start
def generate_ratings(run_id: str, df: pd.DataFrame, client: OpenAI, prompt: str, conn: sqlite3.Connection) -> pd.DataFrame:
    
    check_job_completed("generate_ratings", run_id, conn)

    ratings = []
    for _, row in df.iterrows():
        try:

            if _GENERATE_INTERMEDIATE_SUMMARIES:
                intermediate_prompt = f"Summarize the following news article in a concise manner:\n\n{row['body']}\n\nSummary:"
                intermediate_response = client.responses.create(
                    model=_MODEL,
                    instructions=intermediate_prompt,
                    input=row['body']
                )
                intermediate_summary = intermediate_response.output_text
                final_input = intermediate_summary
            else:
                final_input = row['body']

            response = client.responses.create(
                model=_MODEL,
                instructions=prompt,
                input=final_input
            )
            ratings.append(response.output_text)
        except Exception:
            ratings.append(None)
    df['rating'] = ratings
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df = df[df['rating'].notna()]
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').dropna()

    mark_job_completed("generate_ratings", run_id, conn)

    return df

@print_start
def create_regression_data(df_prices: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    
    df_prices['D_TRDPRC_1'] = df_prices.groupby(['Index', 'start'])['TRDPRC_1'].shift(1) - df_prices['TRDPRC_1']
    # instead of shifting, drop where D_TRDPRC_1     is NA
    df_prices = df_prices[df_prices['D_TRDPRC_1'].notna()]
    data_for_regression = pd.merge(df_prices, ratings, left_on='start', right_on='timestamp', how='inner')
    data_for_regression['Timestamp'] = pd.to_datetime(data_for_regression['Timestamp'], errors='coerce')
    data_for_regression = data_for_regression.drop_duplicates(subset=['timestamp', 'Index']).dropna()

    return data_for_regression

@print_start
def regress(data_for_regression: pd.DataFrame) -> RegressionResultsWrapper:

    X = pd.to_numeric(data_for_regression['rating'], errors='coerce') #.dropna().reset_index(drop=True)
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
    cur_local.execute("SELECT 1 FROM JobCompleted WHERE jobId = ? AND runId = ? LIMIT 1", (job_id, run_id))
    return 1 if cur_local.fetchone() is not None else 0

def mark_job_completed(job_id: str, run_id: str, conn: Connection) -> None:
    cur_local = conn.cursor()
    cur_local.execute(
        "INSERT INTO JobCompleted (jobId, runId, completed_at) VALUES (?, ?, ?)",
        (job_id, run_id, datetime.now())
    )
    conn.commit()