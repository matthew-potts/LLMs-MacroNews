import os, json, glob, time, logging
import pandas as pd
from openai import OpenAI
from src.lib.llm_client import LLMClient

# Configure logger
logger = logging.getLogger('batch_creator')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_jsonl(df: pd.DataFrame, output_file: str, client: LLMClient) -> None:

    with open(f'data/prompt_scale.txt') as f:
        prompt = f.read()

    def batch_row(story, client: LLMClient) -> str:
        # Create batch_row object to transform to jsonl compatible with API
        body_escaped = json.dumps(story['body'])[1:-1]  # Remove outer quotes from all text fields
        headline_escaped = json.dumps(story['headline'])[1:-1]
        prompt_escaped = json.dumps(prompt)[1:-1]

        if client.model == "gpt-4o":
            return f'{{"custom_id": "request-{story["storyId"]}.Timestamp-{str(story.name)}", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "gpt-4o", "messages": [{{"role": "system", "content": "{prompt_escaped}"}},{{"role": "user", "content": "{headline_escaped} {body_escaped}"}}],"max_tokens": 3000}}}}'
        elif client.model == "gpt-5-mini":
            return f'{{"custom_id": "request-{story["storyId"]}.Timestamp-{str(story.name)}", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "gpt-5-mini", "messages": [{{"role": "system", "content": "{prompt_escaped}"}},{{"role": "user", "content": "{headline_escaped} {body_escaped}"}}]}}}}'
    
    df = df.apply(lambda story: batch_row(story, client), axis=1)

    # Convert to JSONL (one JSON object per line)
    with open(output_file, 'w') as f: # ../data/analysis/fed_top_news_stories_2025_batch_request.jsonl
        for row in df:
            # f.write(row['batch_request'] + '\n')
            f.write(str(row) + '\n')
    return

def create_batches(
        client: LLMClient,
        input_batch_file: str,
        batch_size: int,
        out_dir: str = '../data/analysis/batches',
        endpoint: str = '/v1/chat/completions',
        max_concurrent_batches: int = 2
    ) -> list:
    # Read all non-empty lines
    with open(input_batch_file, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]

    total = len(lines)
    if total == 0:
        raise ValueError(f"Input file {input_batch_file} contains no lines")

    num_batches = (total + batch_size - 1) // batch_size
    logger.info('Total requests=%s resulting in %s batches', total, num_batches)
    results: list = []
    pending_to_create: list = []  # entries that have been uploaded and await batch creation
    running: list = []  # currently running batch entries (have batch_id)

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_lines = lines[start:end]

        batch_filename = f"batch_{i+1:03d}.jsonl"
        batch_path = os.path.join(out_dir, batch_filename)

        # Write batch file
        with open(batch_path, 'w') as bf:
            bf.write('\n'.join(batch_lines) + '\n')
        logger.info('Wrote batch file %s (%d lines)', batch_path, len(batch_lines))

        entry: dict = {"batch_file": batch_path}
        # Optionally upload the batch file to the Files API

        logger.info('Uploading %s', batch_path)
        file_resp = client.client.files.create(
            file=open(batch_path, 'rb'),
            purpose='batch',
            expires_after={
                'anchor': 'created_at',
                'seconds': 2592000
            }
        )
        entry['file_response'] = file_resp
        logger.info('Upload response for %s: %s', batch_path, getattr(file_resp, 'id', str(file_resp)) if not isinstance(file_resp, dict) else str(file_resp))

        # Extract file id from response (support dict-like or attr access)
        if isinstance(file_resp, dict):
            file_id = file_resp.get('id') or file_resp.get('file_id')
        else:
            file_id = getattr(file_resp, 'id', None) or getattr(file_resp, 'file_id', None)

        entry['file_id'] = file_id
        logger.info('Extracted file_id=%s for %s', file_id, batch_path)

        # If we're going to create a batch, enqueue for creation (throttled below)
        pending_to_create.append(entry)
        logger.info('Enqueued %s for batch creation (pending queue size=%d)', batch_path, len(pending_to_create))

    # If user requested batch creation, submit them with throttling

    logger.info('Starting batch creation loop; max_concurrent_batches=%s', max_concurrent_batches)
    # Kick off up to `max_concurrent_batches` immediately
    while pending_to_create or running:
        # Submit new batches while we have capacity
        while pending_to_create and len(running) < max_concurrent_batches:
            next_entry = pending_to_create.pop(0)
            file_id = next_entry.get('file_id')
            batch_file = next_entry.get('batch_file')
            if not file_id:
                # If file_id is missing, raise so the user can inspect the upload response
                logger.error('Missing file_id for uploaded file: %s', next_entry)
                raise RuntimeError(f"Missing file_id for uploaded file: {next_entry}")

            logger.info('Creating batch for %s using file_id=%s', batch_file, file_id)
            batch_resp = client.client.batches.create(
                input_file_id=file_id,
                endpoint=endpoint,
                completion_window='24h'
            )
            next_entry['batch_response'] = batch_resp

            # extract batch id (support dict or attr access)
            if isinstance(batch_resp, dict):
                batch_id = batch_resp.get('id') or batch_resp.get('batch_id')
            else:
                batch_id = getattr(batch_resp, 'id', None) or getattr(batch_resp, 'batch_id', None)

            next_entry['batch_id'] = batch_id
            logger.info('Submitted batch_id=%s for %s; running slots=%d', batch_id, batch_file, len(running)+1)
            running.append(next_entry)
            # also record it in results so the caller has the upload + batch info
            results.append(next_entry)

        # If there are no running batches and none pending (done), break
        if not running and not pending_to_create:
            logger.info('No running or pending batches remaining; exiting creation loop')
            break

        # Poll running batches and remove those that have finished (completed/failed)
        to_remove = []
        for r in running:
            bid = r.get('batch_id')
            status = client.client.batches.retrieve(bid).status
            
            logger.info('Batch %s status=%s', bid, status)
            if status in ('completed', 'failed') or status == 'succeeded':  
                to_remove.append(r)

        for r in to_remove:
            running.remove(r)
            logger.info('Batch %s finished; freed a slot (running now=%d)', r.get('batch_id'), len(running))

        if pending_to_create and len(running) < max_concurrent_batches:
            continue

        if running:
            logger.info('Waiting %s seconds before next status poll (running=%d pending=%d)', '10.0', len(running), len(pending_to_create))
            time.sleep(10.0)    

    logger.info('create_batches finished; total results=%d', len(results))
    return results


def write_results_from_batches(results, output_jsonl_path, client: OpenAI) -> None:
    # Ensure output directory exists and start with a clean file
    out_dir = os.path.dirname(output_jsonl_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    # Clear or create the output JSONL
    with open(output_jsonl_path, 'w', encoding='utf-8') as _f:
        pass

    total_written = 0
    for entry in results:
        batch_id = entry.get('batch_id')

        batch_info = client.client.batches.retrieve(batch_id)

        # locate output file id (support different shapes)
        file_id = getattr(batch_info, 'output_file_id', None) or getattr(batch_info, 'output_file', None)

        logger.info('Downloading output file %s for batch %s', file_id, batch_id)
        try: 
            results = client.client.files.content(file_id)
        except Exception as e:
            logger.error('Error downloading output file %s for batch %s: %s', file_id, batch_id, e)
            continue
        
        with open(output_jsonl_path, 'a', encoding='utf-8') as outf:
            for raw in results.iter_lines():
                line = (raw.decode('utf-8', errors='replace') if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()
                if not line:
                    continue                    
                obj = json.loads(line)
                outf.write(json.dumps(obj, ensure_ascii=False) + '\n')
                total_written += 1

    logger.info('Wrote %d JSON lines to %s', total_written, output_jsonl_path)
    return output_jsonl_path


def read_batch_outputs(outputs_dir=None, glob_pattern='*.jsonl') -> 'pd.DataFrame':

    paths = glob.glob(os.path.join(outputs_dir, glob_pattern))

    rows = []
    for p in paths:
        logger.info('Parsing file %s', p)
        with open(p, 'r', encoding='utf-8') as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logger.warning('Skipping malformed JSON line %s:%d (%s)', p, lineno, e)
                    continue

                raw_req = obj.get('response', {}).get('custom_id') or obj.get('custom_id') or obj.get('id')
                # Split request_id into requestID and timestamp (if present)
                requestID = raw_req
                timestamp = None
                if isinstance(raw_req, str) and '.Timestamp' in raw_req:
                    left, right = raw_req.split('.Timestamp', 1)
                    requestID = left
                    timestamp = right.lstrip('-')                
                custom_id = obj.get('custom_id') or obj.get('customId') or None
                story_id = custom_id[len('request-'):]
        
                content = obj['response']['body']['choices'][0]['message']['content']
                rows.append({
                    'requestID': requestID,
                    'timestamp': timestamp,
                    'storyID': story_id,
                    'content': content,
                    'source_file': p
                })

    df = pd.DataFrame(rows)
    logger.info('Parsed %d rows from %d files', len(df), len(paths))
    return df   