import pandas as pd
import json
import argparse
import os
import re
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-CSV to JSONL Converter for Shiva-DiT Dataset")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input CSV file paths (space separated)")
    parser.add_argument("--output", type=str, default="processed_dataset.jsonl", help="Output JSONL file path")
    parser.add_argument("--min_length", type=int, default=15, help="Minimum text length to keep (default: 15)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of records to output (optional)")
    return parser.parse_args()

def clean_text_content(text):
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"').strip("'")
    text = text.replace('\\"', '"').replace("\\'", "'")
    
    return text.strip()

def is_high_quality(text, min_len=15):
    if not text:
        return False
    
    if len(text) < min_len:
        return False
        
    if text.isdigit():
        return False
    
    bad_keywords = ["test", "测试", "default", "null", "undefined"]
    if text.lower() in bad_keywords:
        return False

    clean_chars = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    if len(clean_chars) < 3:
        return False

    return True

def main():
    args = parse_args()
    
    all_dfs = []
    print(f"Received {len(args.input)} input file(s).")
    
    for file_path in args.input:
        if not os.path.exists(file_path):
            print(f"[Warn] File not found, skipping: {file_path}")
            continue
            
        try:
            print(f"Reading {file_path}...")
            df_temp = pd.read_csv(file_path, on_bad_lines='skip')
            
            if 'prompt' in df_temp.columns:
                all_dfs.append(df_temp[['prompt']])
            else:
                print(f"[Warn] Column 'prompt' not found in {file_path}, skipping.")
        except Exception as e:
            print(f"[Error] Failed to read {file_path}: {e}")

    if not all_dfs:
        print("No valid data loaded. Exiting.")
        return

    print("Merging datasets...")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total raw records: {len(df)}")
    
    tqdm.pandas(desc="Processing")

    print("Cleaning text...")
    df = df.dropna(subset=['prompt'])
    df['clean_text'] = df['prompt'].astype(str).progress_apply(clean_text_content)
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['clean_text'])
    print(f"Deduplicated: {before_dedup} -> {len(df)} records")

    print(f"Filtering high quality texts (min_len={args.min_length})...")
    tqdm.pandas(desc="Filtering")
    df = df[df['clean_text'].progress_apply(lambda x: is_high_quality(x, args.min_length))]

    print("Sorting by length (longest first)...")
    df['length'] = df['clean_text'].str.len()
    df = df.sort_values(by='length', ascending=False)

    if args.limit:
        print(f"Limiting to top {args.limit} records.")
        df = df.head(args.limit)

    final_count = len(df)
    print(f"Final valid count: {final_count}")

    print(f"Writing to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=final_count, desc="Writing"):
            item = {
                "id": str(idx),           
                "text": row['clean_text'],
                "length": row['length']
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()