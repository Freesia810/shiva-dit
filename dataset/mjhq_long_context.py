import json
import base64
import io
import time
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm


try:
    import whale
    from whale import TextGeneration
except ImportError:
    print("Warning: 'whale' library not found.")

DEFAULT_INPUT_JSONL = "../dataset/MJHQ-30K/test.jsonl"
DEFAULT_IMAGE_ROOT = "../dataset/MJHQ-30K/"
DEFAULT_OUTPUT_JSONL = "mjhq_dense_captions.jsonl"
MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking"
MAX_QPS = 2.0
NUM_THREADS = 5

class RateLimiter:
    def __init__(self, max_qps):
        self.interval = 1.0 / max_qps
        self.last_call_times = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            self.last_call_times = [t for t in self.last_call_times if current_time - t < 1.0]
            
            if len(self.last_call_times) > 0:
                elapsed = current_time - self.last_call_times[-1]
                if elapsed < self.interval:
                    time.sleep(self.interval - elapsed)
            
            self.last_call_times.append(time.time())

def pil_image_to_base64_url(image, fmt="JPEG"):
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format=fmt)
    img_byte_data = buffered.getvalue()
    base64_str = base64.b64encode(img_byte_data).decode('utf-8')
    mime_type = f"image/{fmt.lower()}"
    return f"data:{mime_type};base64,{base64_str}"

def process_single_item(item, image_root, rate_limiter):
    file_name = item.get('file_name') or item.get('image')
    if not file_name:
        return None, "No filename found"

    image_path = os.path.join(image_root, file_name)
    
    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"

    try:
        with Image.open(image_path) as image:
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image_url_str = pil_image_to_base64_url(image, fmt="JPEG")

        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an expert dataset annotator specializing in preparing high-quality training data for next-generation text-to-image models (like Flux). "
                    "Your output will be used as the 'Ground Truth' caption to teach the model how to synthesize images."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url_str,
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Generate a dense, comprehensive caption for this image to be used as training data. "
                            "The goal is to describe the image so precisely that a generative model could reconstruct it solely from your text without seeing the original. "
                            
                            "**Requirements:**\n"
                            "1. **Length:** Aim for 350-450 words. Fill the context window with visual information.\n"
                            "2. **Format:** Use fluent, complex English sentences. **DO NOT** use comma-separated tags or lists (e.g., avoid '4k, masterpiece, solo'). Write like a novelist describing a scene.\n"
                            "3. **Coverage:** Systematically describe:\n"
                            "   - The main subject (anatomy, pose, clothing textures, facial features).\n"
                            "   - The environment (spatial layout, background objects, depth of field).\n"
                            "   - Lighting (direction, color temperature, shadows, subsurface scattering).\n"
                            "   - Artistic style (medium, brushstrokes, color grading, film grain).\n"
                            "4. **Density:** Do not waste tokens on intro phrases like 'This image shows'. Start directly with the visual description."
                        )
                    }
                ]
            }
        ]

        extend_fields = {"top_k": 1}

        rate_limiter.wait()

        response = TextGeneration.chat(
            model=MODEL_NAME,
            messages=msgs,
            stream=False,
            temperature=1.0,
            timeout=180,
            max_tokens=2000,
            top_p=0.8,
            extend_fields=extend_fields
        )

        if response._error_code is not None:
            return None, f"API Error {response._error_code}: {response._message}"
        
        caption = response.choices[0].message.content
        
        result_item = item.copy()
        result_item['dense_caption'] = caption
        return result_item, None

    except Exception as e:
        return None, f"Exception: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Brush MJHQ dataset with Qwen-VL")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_JSONL, help="Input metadata JSONL file")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT, help="Root directory for images")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL file")
    parser.add_argument("--qps", type=float, default=MAX_QPS, help="Max requests per second")
    parser.add_argument("--workers", type=int, default=NUM_THREADS, help="Number of threads")
    
    args = parser.parse_args()

    api_key = os.environ.get("WHALE_API_KEY")
    base_url = os.environ.get("WHALE_BASE_URL")

    if not api_key or not base_url:
        print("Error: Please set WHALE_API_KEY and WHALE_BASE_URL environment variables.")
        sys.exit(1)

    TextGeneration.set_api_key(api_key, base_url=base_url)

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Reading input from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        all_items = [json.loads(line) for line in f]
    
    processed_files = set()
    if os.path.exists(args.output):
        print(f"Checking existing output {args.output} for resume...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'file_name' in data:
                        processed_files.add(data['file_name'])
                    elif 'image' in data:
                         processed_files.add(data['image'])
                except:
                    pass
    
    print(f"Found {len(processed_files)} already processed images.")
    
    items_to_process = [
        item for item in all_items 
        if item.get('file_name', item.get('image')) not in processed_files
    ]
    
    print(f"Remaining images to process: {len(items_to_process)}")
    if len(items_to_process) == 0:
        print("All done!")
        return

    rate_limiter = RateLimiter(args.qps)
    file_lock = threading.Lock()

    print(f"Starting processing with {args.workers} workers, Max QPS: {args.qps}...")
    
    with open(args.output, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {
                executor.submit(process_single_item, item, args.image_root, rate_limiter): item 
                for item in items_to_process
            }

            for future in tqdm(as_completed(future_to_item), total=len(items_to_process)):
                item = future_to_item[future]
                result, error = future.result()

                if result:
                    with file_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                else:
                    tqdm.write(f"Failed to process {item.get('file_name', 'unknown')}: {error}")

    print("Processing finished.")

if __name__ == "__main__":
    main()