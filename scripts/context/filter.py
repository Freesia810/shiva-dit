import json
import argparse
import os

def process_jsonl(input_path, output_path, token_limit, ratio):
    # Calculate word threshold based on token limit and ratio
    # word_count = token_count / ratio
    word_threshold = int(token_limit / ratio)
    
    count_total = 0
    count_saved = 0

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            count_total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract dense_caption as the new text field
            dense_content = data.get('dense_caption', "")
            words = dense_content.split()
            current_word_count = len(words)
            
            # Filter based on converted word threshold
            if current_word_count >= word_threshold:
                new_item = {
                    "file_name": data.get("file_name", ""),
                    "text": dense_content
                }
                fout.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                count_saved += 1

    print(f"--- Processing Statistics ---")
    print(f"Total processed: {count_total}")
    print(f"Token limit set: {token_limit}")
    print(f"Word threshold (ratio {ratio}): {word_threshold}")
    print(f"Samples saved: {count_saved}")
    print(f"Output path: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter jsonl by token count threshold")
    
    parser.add_argument("--input", type=str, required=True, help="Input jsonl file path")
    parser.add_argument("--output", type=str, default="output.jsonl", help="Output jsonl file path")
    parser.add_argument("--tokens", type=int, default=512, help="Token count threshold (e.g., 512, 1024)")
    parser.add_argument("--ratio", type=float, default=1.3, help="Token-to-word conversion ratio")
    
    args = parser.parse_args()
    
    process_jsonl(args.input, args.output, args.tokens, args.ratio)