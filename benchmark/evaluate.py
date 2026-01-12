import torch
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from types import ModuleType
import warnings
import open_clip
import io
import base64

try:
    import whale
    from whale import TextGeneration
    WHALE_AVAILABLE = True
except ImportError:
    WHALE_AVAILABLE = False

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchmetrics.multimodal import CLIPImageQualityAssessment
    import torchvision.transforms.functional as TF
except ImportError:
    print(
        "Error: Required libraries missing. Please run:\n"
        "pip install pytorch-fid torchmetrics[image]"
    )
    sys.exit(1)

# Local BERT path handling
LOCAL_BERT_PATH = "../bert-base-uncased"
from transformers import BertTokenizer, BertModel

_original_tokenizer_from_pretrained = BertTokenizer.from_pretrained
_original_model_from_pretrained = BertModel.from_pretrained

def _patched_tokenizer_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased" and os.path.exists(LOCAL_BERT_PATH):
        pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

def _patched_model_loader(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "bert-base-uncased" and os.path.exists(LOCAL_BERT_PATH):
        pretrained_model_name_or_path = LOCAL_BERT_PATH
    return _original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

BertTokenizer.from_pretrained = classmethod(_patched_tokenizer_loader)
BertModel.from_pretrained = classmethod(_patched_model_loader)

# ImageReward compatibility
sys.modules["ImageReward.ReFL"] = ModuleType("ImageReward.ReFL")

import transformers.modeling_utils
import transformers.pytorch_utils

if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    if hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
        transformers.modeling_utils.apply_chunking_to_forward = (
            transformers.pytorch_utils.apply_chunking_to_forward
        )
if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
    if hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        transformers.modeling_utils.find_pruneable_heads_and_indices = (
            transformers.pytorch_utils.find_pruneable_heads_and_indices
        )
if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
    if hasattr(transformers.pytorch_utils, "prune_linear_layer"):
        transformers.modeling_utils.prune_linear_layer = (
            transformers.pytorch_utils.prune_linear_layer
        )

# Optional imports for advanced metrics
try:
    import ImageReward as RM
except ImportError:
    RM = None
    print("Warning: ImageReward not installed. IR score will be skipped.")

class QwenVLClient:
    def __init__(self, model_name="Qwen3-VL-235B-A22B-Thinking"):
        if not WHALE_AVAILABLE:
            raise ImportError("Whale library is not installed.")
        
        self.model_name = model_name
        self.api_key = os.environ.get("WHALE_API_KEY")
        self.base_url = os.environ.get("WHALE_BASE_URL")
        
        if self.api_key and self.base_url:
            TextGeneration.set_api_key(self.api_key, base_url=self.base_url)
        else:
            print("Warning: WHALE_API_KEY or WHALE_BASE_URL not set.")

    def _pil_to_base64(self, image: Image.Image, fmt="JPEG") -> str:
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format=fmt)
        img_byte_data = buffered.getvalue()
        base64_str = base64.b64encode(img_byte_data).decode('utf-8')
        mime_type = f"image/{fmt.lower()}"
        return f"data:{mime_type};base64,{base64_str}"

    def ask(self, image: Image.Image, question: str) -> str:
        image_url = self._pil_to_base64(image)
        
        msgs = [
            {
                "role": "system",
                "content": "You are a helpful assistant for evaluating image generation quality."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        try:
            response = TextGeneration.chat(
                model=self.model_name,
                messages=msgs,
                stream=False,
                temperature=1.0,
                max_tokens=2000,
                top_p=0.8,
                extend_fields={"top_k": 1}
            )
            
            if response._error_code is not None:
                print(f"API Error: {response._message}")
                return ""
            
            content = (response.choices[0].message.to_dict())['content']            
            if content is None:
                raise ValueError("Qwen API returned no content.")
            
            return content.strip()
            
        except Exception as e:
            print(f"Qwen API Exception: {e}")
            return ""

class LocalLLaVAClient:
    def __init__(self, model_path, device):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not installed.")
            
        print(f"Loading Local LLaVA from {model_path}...")
        self.device = device
        self.dtype = torch.float16
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading LLaVA: {e}")
            raise e

    def ask(self, image: Image.Image, question: str) -> str:
        prompt_text = f"USER: <image>\n{question}\nASSISTANT:"
        
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(self.device, self.dtype)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0
            )
        
        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()
        
        return generated_text

class LongTextEvaluator:
    """Comprehensive evaluator for text-to-image generation with focus on long text."""
    
    def __init__(self, device, config=None):
        self.device = device
        self.config = config or {}
        print(f"Initializing evaluator on {device}...")

        self.clip_score = None
        self.clip_iqa = None
        self.ir_model = None
        self.hpsv2_model = None
        self.hpsv2_preprocess = None
        self.hpsv2_tokenizer = None
        self.pick_model = None
        self.vqa_client = None
        self.tifa_enabled = False
        
        # Initialize traditional metrics
        if self.config['skip_traditional']:
            print("✓ Skipping traditional metrics")
        else:
            print("✓ Loading traditional metrics")
            self._init_traditional_metrics()
        
        # Initialize long-text specific metrics
        if self.config['skip_long_text']:
            print("✓ Skipping long-text metrics")
        else:
            print("✓ Loading long-text metrics")
            self._init_long_text_metrics()
        
        # Results storage
        self.results = {
            "clip_scores": [],
            "clip_iqa_scores": [],
            "ir_scores": [],
            "hpsv2_scores": [],
            "pick_scores": [],
            "vqa_scores": [],
            "tifa_scores": []
        }
    
    def _init_traditional_metrics(self):
        """Initialize traditional evaluation metrics."""
        try:
            clip_model_path = self.config.get("clip_model_path", "../openai-clip-vit-base-patch16")
            self.clip_score = CLIPScore(model_name_or_path=clip_model_path).to(self.device)
            self.clip_iqa = CLIPImageQualityAssessment(model_name_or_path=clip_model_path).to(self.device)
            print("✓ Traditional CLIP metrics loaded")
        except Exception as e:
            print(f"Warning: Failed to load CLIP metrics: {e}")
        
        # ImageReward
        if RM:
            try:
                ir_model_path = self.config.get("ir_model_path", "../ImageReward/ImageReward.pt")
                ir_config_path = self.config.get("ir_config_path", "../ImageReward/med_config.json")
                self.ir_model = RM.load(ir_model_path, device=self.device, med_config=ir_config_path)
                print("✓ ImageReward model loaded")
            except Exception as e:
                print(f"Warning: Failed to load ImageReward: {e}")
    
    def _init_long_text_metrics(self):
        """Initialize metrics specifically designed for long text evaluation."""
        
        # HPSv2 - Human Preference Score v2
        try:
            print("Loading HPSv2 model...")
            hpsv2_pt_path = self.config.get("hpsv2_model_path", "../HPSv2/HPS_v2.1_compressed.pt")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', 
                pretrained=None,
                precision='amp'
            )
            
            if os.path.exists(hpsv2_pt_path):
                print(f"Loading weights from: {hpsv2_pt_path}")
                checkpoint = torch.load(hpsv2_pt_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict)
            else:
                print(f"Warning: HPSv2 .pt file not found at {hpsv2_pt_path}, model will be random!")

            self.hpsv2_model = model.to(self.device)
            self.hpsv2_model.eval()
            self.hpsv2_preprocess = preprocess
            self.hpsv2_tokenizer = open_clip.get_tokenizer('ViT-H-14')
            print("✓ HPSv2 model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load HPSv2: {e}")
        
        # PickScore
        self.pick_model = None
        try:
            from transformers import AutoProcessor, AutoModel
            pick_model_path = self.config.get("pick_model_path", "yuvalkirstain/PickScore_v1")
            self.pick_processor = AutoProcessor.from_pretrained(pick_model_path)
            self.pick_model = AutoModel.from_pretrained(pick_model_path).to(self.device)
            self.pick_model.eval()
            print("✓ PickScore model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load PickScore: {e}")
        
        backend_type = self.config.get("vqa_backend", "local")
        if backend_type == "api":
            print("Initializing Qwen-VL API Client...")
            try:
                self.vqa_client = QwenVLClient()
                print("✓ Qwen-VL API ready.")
            except Exception as e:
                print(f"❌ Failed to init API client: {e}")
        elif backend_type == "local":
            print("Initializing Local LLaVA Client...")
            model_path = self.config.get("vqa_model_path", "llava-hf/llava-1.5-13b-hf")
            if os.path.exists(model_path):
                try:
                    self.vqa_client = LocalLLaVAClient(model_path, self.device)
                    print(f"✓ Local LLaVA loaded from {model_path}")
                except Exception as e:
                    print(f"❌ Failed to load local LLaVA: {e}")
            else:
                print(f"❌ Model path not found: {model_path}")
        
        self.llm_model = None
        self.llm_tokenizer = None
        
        try:
            print("Loading Llama-3 for TIFA text decomposition...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            llm_path = self.config.get("llm_model_path", "meta-llama/Meta-Llama-3-8B-Instruct")
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            print("✓ Llama-3 loaded successfully for text segmentation")
        except Exception as e:
            print(f"Warning: Failed to load LLM for TIFA: {e}")
        
        # TIFA-style evaluation
        self.tifa_enabled = (self.vqa_client is not None) and (self.llm_model is not None)
        if self.tifa_enabled:
            print("✓ TIFA-style evaluation enabled (VQA + LLM)")
        else:
            print("Warning: TIFA disabled (Missing VQA client or LLM model)")
        
    def preprocess_image(self, image):
        """Preprocess image for evaluation."""
        return TF.to_tensor(image).to(self.device).unsqueeze(0)
    
    def preprocess_image_uint8(self, image):
        """Preprocess image to uint8 format."""
        t = TF.to_tensor(image).to(self.device) * 255
        return t.to(torch.uint8).unsqueeze(0)
    
    @torch.no_grad()
    def evaluate_traditional_metrics(self, image_pil, prompt):
        """Evaluate traditional metrics (CLIP, ImageReward)."""
        if self.clip_score is None:
            return
        
        gen_tensor = self.preprocess_image(image_pil)
        gen_uint8 = self.preprocess_image_uint8(image_pil)
        
        # CLIP Score
        if prompt:
            try:
                score = self.clip_score(gen_uint8, [prompt])
                self.results["clip_scores"].append(score.item())
            except Exception as e:
                print(f"Warning: CLIP Score failed: {e}")
        
        # CLIP IQA
        try:
            iqa_out = self.clip_iqa(gen_tensor)
            if isinstance(iqa_out, dict):
                mean_iqa = torch.stack(list(iqa_out.values())).mean().item()
                self.results["clip_iqa_scores"].append(mean_iqa)
            else:
                self.results["clip_iqa_scores"].append(iqa_out.item())
        except Exception as e:
            print(f"Warning: CLIP IQA failed: {e}")
        
        # ImageReward
        if self.ir_model and prompt:
            try:
                ir_score = self.ir_model.score(prompt, image_pil)
                self.results["ir_scores"].append(ir_score)
            except Exception as e:
                print(f"Warning: ImageReward failed: {e}")
    
    @torch.no_grad()
    def evaluate_hpsv2(self, image_pil, prompt):
        if self.hpsv2_model is None or not prompt:
            return
        
        try:
            image = self.hpsv2_preprocess(image_pil).unsqueeze(0).to(self.device)
            
            text = self.hpsv2_tokenizer([prompt]).to(self.device)
            
            with torch.cuda.amp.autocast() if self.device != 'cpu' else torch.autocast('cpu'):
                image_features = self.hpsv2_model.encode_image(image)
                text_features = self.hpsv2_model.encode_text(text)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                hps_score = (image_features @ text_features.T).item()
            
            self.results["hpsv2_scores"].append(hps_score)
        except Exception as e:
            print(f"Warning: HPSv2 evaluation failed: {e}")
    
    @torch.no_grad()
    def evaluate_pick_score(self, image_pil, prompt):
        """Evaluate PickScore."""
        if self.pick_model is None or not prompt:
            return
        
        try:
            inputs = self.pick_processor(text=prompt, images=image_pil, return_tensors="pt").to(self.device)
            outputs = self.pick_model(**inputs)
            score = outputs.logits_per_image.squeeze().cpu().item()
            self.results["pick_scores"].append(score)
        except Exception as e:
            print(f"Warning: PickScore evaluation failed: {e}")
    
    @torch.no_grad()
    def evaluate_vqa_score(self, image_pil, prompt):
        if self.vqa_client is None: return

        question = (
            f"Please rate how well this image matches the description: '{prompt}'. "
            "Give a score from 0 to 100, where 0 is completely irrelevant and 100 is a perfect match. "
            "Answer with the number only."
        )
        
        try:
            answer = self.vqa_client.ask(image_pil, question)
            import re
            digits = re.findall(r"\d+", answer)
            if digits:
                score = float(digits[0]) / 100.0
                self.results["vqa_scores"].append(score)
            else:
                print(f"VQA format error, raw answer: {answer}")
        except Exception as e:
            print(f"VQA Eval Error: {e}")
    
    def _extract_elements_with_llama(self, long_prompt):
        if not self.llm_model:
            return [long_prompt]

        system_msg = (
            "You are an expert AI visual assistant. "
            "Your task is to extract all visual entities, objects, attributes, and actions from the text. "
            "Break down complex sentences into simple, atomic visual phrases.\n"
            "Format: Output each element on a new line starting with a dash '- '."
        )
        user_msg = f"Extract visual elements from this description:\n\n{long_prompt}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        input_ids = self.llm_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.llm_model.device)

        terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.llm_model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.0
            )
        
        response = outputs[0][input_ids.shape[-1]:]
        decoded_output = self.llm_tokenizer.decode(response, skip_special_tokens=True)
        
        elements = []
        for line in decoded_output.split('\n'):
            line = line.strip()
            if line.startswith("- "):
                clean_elem = line[2:].strip()
                if len(clean_elem) > 2:
                    elements.append(clean_elem)
        
        if not elements:
            print("Warning: LLM extraction returned empty, falling back to simple split.")
            elements = [s.strip() for s in long_prompt.split(',') if len(s.strip()) > 5]
            
        return elements

    @torch.no_grad()
    def evaluate_tifa_score(self, image_pil, prompt):
        if not self.tifa_enabled: return

        elements = self._extract_elements_with_llama(prompt)
            
        scores = []
        for elem in elements: 
            question = f"Is the following element present in the image: '{elem}'? Answer Yes or No."
            
            try:
                answer = self.vqa_client.ask(image_pil, question).lower()

                if "yes" in answer:
                    scores.append(1.0)
                elif "no" in answer:
                    scores.append(0.0)
                else:
                    pass 
            except Exception as e:
                print(f"TIFA VQA Error for '{elem}': {e}")
                pass
        
        if scores:
            final_score = sum(scores) / len(scores)
            self.results["tifa_scores"].append(final_score)
    
    def evaluate_single_image(self, image_pil, prompt):
        """Evaluate a single image with all available metrics."""
        self.evaluate_traditional_metrics(image_pil, prompt)
        self.evaluate_hpsv2(image_pil, prompt)
        self.evaluate_pick_score(image_pil, prompt)
        self.evaluate_vqa_score(image_pil, prompt)
        self.evaluate_tifa_score(image_pil, prompt)
    
    def compute_final_metrics(self, generated_images_dir, fid_stats_path=None):
        """Compute aggregated metrics."""
        print("\nComputing final metrics...")
        metrics = {}
        
        # FID calculation
        if fid_stats_path and os.path.exists(fid_stats_path) and os.path.exists(generated_images_dir):
            print(f"Computing FID between {generated_images_dir} and {fid_stats_path}")
            try:
                fid_value = calculate_fid_given_paths(
                    paths=[generated_images_dir, fid_stats_path],
                    batch_size=50,
                    device=self.device,
                    dims=2048
                )
                metrics["FID"] = fid_value
            except Exception as e:
                print(f"Error computing FID: {e}")
                metrics["FID"] = -1.0
        else:
            print("Skipping FID (missing reference stats or output directory)")
            metrics["FID"] = -1.0
        
        # Aggregate other metrics
        for key, scores in self.results.items():
            if scores:
                metric_name = key.replace("_scores", "").upper()
                if metric_name == "CLIP":
                    metric_name = "CLIP_SCORE"
                elif metric_name == "CLIP_IQA":
                    metric_name = "CLIP_IQA"
                elif metric_name == "IR":
                    metric_name = "IMAGE_REWARD"
                
                metrics[metric_name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "count": len(scores)
                }
        
        return metrics


def load_image_prompt_pairs(images_dir, jsonl_path=None, metadata_path=None):
    """Load image-prompt pairs for evaluation."""
    pairs = []
    
    # Try to load from generation metadata first
    if metadata_path and os.path.exists(metadata_path):
        print(f"Loading from generation metadata: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for item in metadata:
            image_path = os.path.join(images_dir, item["file_name"])
            if os.path.exists(image_path):
                pairs.append({
                    "image_path": image_path,
                    "prompt": item["prompt"],
                    "metadata": item
                })
    
    # Fallback: load from JSONL and match by filename
    elif jsonl_path and os.path.exists(jsonl_path):
        print(f"Loading from JSONL: {jsonl_path}")
        prompts_by_index = {}
        
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if "text" in item:
                    prompts_by_index[i] = item["text"]
        
        # Match image files to prompts
        for filename in os.listdir(images_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Extract global index from filename (format: p{global_idx:04d}_s{j:02d}.png)
                if filename.startswith('p') and '_s' in filename:
                    try:
                        global_idx = int(filename.split('_')[0][1:])
                        if global_idx in prompts_by_index:
                            pairs.append({
                                "image_path": os.path.join(images_dir, filename),
                                "prompt": prompts_by_index[global_idx],
                                "metadata": {"global_index": global_idx, "file_name": filename}
                            })
                    except ValueError:
                        continue
    
    # Last resort: just load images without prompts
    else:
        print("Warning: No prompt source found. Evaluating images without text alignment metrics.")
        for filename in os.listdir(images_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                pairs.append({
                    "image_path": os.path.join(images_dir, filename),
                    "prompt": None,
                    "metadata": {"file_name": filename}
                })
    
    print(f"Loaded {len(pairs)} image-prompt pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Image Evaluation Script")
    
    # Input/Output
    parser.add_argument("--images_dir", type=str, required=True, 
                       help="Directory containing generated images")
    parser.add_argument("--jsonl_path", type=str, default=None,
                       help="Path to original prompts JSONL file")
    parser.add_argument("--metadata_path", type=str, default=None,
                       help="Path to generation metadata JSON file")
    parser.add_argument("--fid_stats", type=str, default=None,
                       help="Path to precomputed FID statistics (.npz file)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save evaluation results")
    
    # Model paths
    parser.add_argument("--clip_model_path", type=str, default="../openai-clip-vit-base-patch16")
    parser.add_argument("--hpsv2_model_path", type=str, default="../HPSv2/HPS_v2.1_compressed.pt")
    parser.add_argument("--pick_model_path", type=str, default="../PickScore_v1")
    parser.add_argument("--vqa_backend", type=str, default="local")
    parser.add_argument("--vqa_model_path", type=str, default="../llava-1.5-13b-hf")
    parser.add_argument("--llm_model_path", type=str, default="../Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ir_model_path", type=str, default="../ImageReward/ImageReward.pt")
    parser.add_argument("--ir_config_path", type=str, default="../ImageReward/med_config.json")
    
    # Evaluation options
    parser.add_argument("--skip_traditional", action="store_true", 
                       help="Skip traditional metrics (CLIP, ImageReward)")
    parser.add_argument("--skip_long_text", action="store_true",
                       help="Skip long-text specific metrics")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to evaluate")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare config
    config = {
        "clip_model_path": args.clip_model_path,
        "hpsv2_model_path": args.hpsv2_model_path,
        "pick_model_path": args.pick_model_path,
        "vqa_backend": args.vqa_backend,
        "vqa_model_path": args.vqa_model_path,
        "llm_model_path": args.llm_model_path,
        "ir_model_path": args.ir_model_path,
        "ir_config_path": args.ir_config_path,
        "skip_traditional": args.skip_traditional,
        "skip_long_text": args.skip_long_text
    }
    
    # Initialize evaluator
    evaluator = LongTextEvaluator(device, config)
    
    # Load image-prompt pairs
    metadata_path = args.metadata_path or os.path.join(args.images_dir, "generation_metadata.json")
    pairs = load_image_prompt_pairs(args.images_dir, args.jsonl_path, metadata_path)
    
    if args.limit:
        pairs = pairs[:args.limit]
    
    # Evaluate images
    print(f"\nEvaluating {len(pairs)} images...")
    for pair in tqdm(pairs, desc="Evaluating"):
        try:
            image = Image.open(pair["image_path"]).convert("RGB")
            evaluator.evaluate_single_image(image, pair["prompt"])
        except Exception as e:
            print(f"Error evaluating {pair['image_path']}: {e}")
            continue
    
    # Compute final metrics
    final_metrics = evaluator.compute_final_metrics(args.images_dir, args.fid_stats)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, dict):
            print(f"{metric_name:<20}: {metric_value['mean']:.4f} ± {metric_value['std']:.4f} (n={metric_value['count']})")
        else:
            print(f"{metric_name:<20}: {metric_value:.4f}")
    
    print("="*60)
    
    # Save results
    output_path = args.output_path or os.path.join(args.images_dir, "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
