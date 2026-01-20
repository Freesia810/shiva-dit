from typing import Any, Callable, Dict, Tuple
import torch
from torch.utils.data import Dataset
import os
import json
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    print(
        "Warning: 'datasets' library not found. HF dataset loading (hf://) will not work."
    )

try:
    from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
        ASPECT_RATIO_1024_BIN,
    )
except ImportError:
    print("Warning: Could not import ASPECT_RATIO_1024_BIN. PixArt-sigma may not work.")
    ASPECT_RATIO_1024_BIN = None


class BaseItemProcessor:
    def __init__(
        self,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        image_dir: str = None,
        pipe: Any = None,
        **kwargs,
    ):
        self.image_dir = image_dir
        self.tokenizers = tokenizers
        self.image_processor = image_processor
        self.pipe = pipe
        self.hf_datasets_cache = {}

    def _load_hf_dataset_split(self, dataset_name: str, split: str):
        local_path = self.image_dir
        
        try:
            if local_path:
                print(f"[Dataset] Found local path for {dataset_name}: {local_path}")
                try:
                    return load_dataset(local_path, split=split)
                except Exception:
                    print(f"[Dataset] load_dataset failed: {local_path}")
            else:
                print(f"[Dataset] No local path found in {self.image_dir}, trying HF Hub: {dataset_name}...")
                return load_dataset(dataset_name, split=split)

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name} (split={split}). Local path found: {local_path}. Error: {e}")

    def preload_hf_dataset(self, dataset_name: str, split: str):
        cache_key = f"{dataset_name}/{split}"
        if cache_key not in self.hf_datasets_cache:
            try:
                self.hf_datasets_cache[cache_key] = self._load_hf_dataset_split(dataset_name, split)
                print(f"[Dataset] Successfully preloaded: {dataset_name} [{split}]")
            except Exception as e:
                print(f"[Dataset] Warning: Failed to preload {dataset_name}: {e}")

    def _load_image_entry(self, entry: Any) -> Image.Image:
        """
        Robust image loader supporting paths (rel/abs), URLs, HF URIs, and PIL Images.
        HF URI format: hf://dataset_name/split/index/column_name
        """
        if isinstance(entry, Image.Image):
            return entry

        if isinstance(entry, str):
            # 0. Check for HF Dataset URI
            if entry.startswith("hf://"):
                if load_dataset is None:
                    raise ImportError(
                        "The 'datasets' library is required to load 'hf://' URIs."
                    )

                # Parse URI: hf://dataset_name/split/index/column_name
                # Example: hf://osu-nlp/MagicBrush/train/123/source_img
                try:
                    content = entry[5:]  # Strip 'hf://'
                    parts = content.split("/")

                    # We assume the last 3 parts are split/index/column
                    # The rest at the beginning is the dataset name
                    if len(parts) < 4:
                        raise ValueError(f"Invalid HF URI format: {entry}")

                    col_name = parts[-1]
                    row_idx = int(parts[-2])
                    split = parts[-3]
                    dataset_name = "/".join(parts[:-3])

                    # Cache key for the dataset split
                    cache_key = f"{dataset_name}/{split}"

                    if cache_key not in self.hf_datasets_cache:
                        raise ValueError(f"Invalid HF URI format: {entry}")

                    dataset = self.hf_datasets_cache[cache_key]
                    image = dataset[row_idx][col_name]

                    if not isinstance(image, Image.Image):
                        # In case the column is a path string, recursive load
                        return self._load_image_entry(image)

                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image

                except Exception as e:
                    raise ValueError(f"Failed to load HF URI '{entry}': {e}")

            # 1. Check if absolute path exists
            if os.path.exists(entry):
                return load_image(entry)

            # 2. Check relative to image_dir
            if self.image_dir:
                full_path = os.path.join(self.image_dir, entry)
                if os.path.exists(full_path):
                    return load_image(full_path)

                # Try looking in parent dir (common issue with dataset structures)
                parent_path = os.path.join(os.path.dirname(self.image_dir), entry)
                if os.path.exists(parent_path):
                    return load_image(parent_path)

            # 3. Try loading as URL or fallback (load_image handles URLs)
            try:
                return load_image(entry)
            except Exception:
                pass

        raise FileNotFoundError(f"Could not load image from entry: {entry}")

    def _load_and_process_image(
        self, item: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Use "file_name" by default, but fall back if item is different
        image_entry = item.get("file_name") or item.get("image")
        if image_entry is None:
            raise ValueError(f"Item missing 'file_name' or 'image' key: {item.keys()}")

        image = self._load_image_entry(image_entry)
        original_size = (image.height, image.width)
        processed_image = self.image_processor.preprocess(image)
        return processed_image, original_size

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FluxDevProcessor(BaseItemProcessor):
    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        processed_image, original_size = self._load_and_process_image(item)
        prompt = item["text"]

        # (clip, t5)
        input_ids = tuple(
            tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length or 512,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            for tokenizer in self.tokenizers
        )

        return {
            "pixel_values": processed_image,
            "input_ids": input_ids,
            "original_size": torch.tensor(original_size),
        }


class PixArtSigmaProcessor(BaseItemProcessor):
    def _load_and_process_image(
        self, item: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Override to add specialized resizing logic
        image_entry = item.get("file_name") or item.get("image")
        image = self._load_image_entry(image_entry)

        original_size = (image.height, image.width)
        processed_image = self.image_processor.preprocess(image)

        if ASPECT_RATIO_1024_BIN is None:
            raise ImportError(
                "ASPECT_RATIO_1024_BIN not available for PixArtSigmaProcessor"
            )

        height, width = self.image_processor.classify_height_width_bin(
            original_size[0], original_size[1], ASPECT_RATIO_1024_BIN
        )
        processed_image = self.image_processor.resize_and_crop_tensor(
            processed_image, height, width
        )
        return processed_image, original_size

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipe is None:
            raise ValueError(
                "PixArtSigmaProcessor requires 'pipe' to be passed for text preprocessing."
            )

        processed_image, original_size = self._load_and_process_image(item)
        prompt = self.pipe._text_preprocessing(item["text"], clean_caption=True)

        max_sequence_length = 300
        text_inputs = self.tokenizers[0](
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = (text_inputs.input_ids[0],)
        attention_mask = (text_inputs.attention_mask[0],)

        return {
            "pixel_values": processed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_size": torch.tensor(original_size),
        }


class SD3Processor(BaseItemProcessor):
    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if len(self.tokenizers) != 3:
            raise ValueError("SD3Processor requires 3 tokenizers")

        processed_image, original_size = self._load_and_process_image(item)
        prompt = item["text"]

        clip_tokenizer_max_length = self.tokenizers[0].model_max_length or 77
        t5_tokenizer_max_length = 256

        # (clip, clip, t5)
        input_ids = [
            tokenizer(
                prompt,
                padding="max_length",
                max_length=clip_tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            for tokenizer in self.tokenizers[:2]
        ]
        input_ids.append(
            self.tokenizers[2](
                prompt,
                padding="max_length",
                max_length=t5_tokenizer_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids[0]
        )

        return {
            "pixel_values": processed_image,
            "input_ids": tuple(input_ids),
            "original_size": torch.tensor(original_size),
        }


class QwenImageProcessor(BaseItemProcessor):
    def __init__(
        self,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        image_dir: str = None,
        pipe: Any = None,
        **kwargs,
    ):
        super().__init__(
            image_dir=image_dir,
            tokenizers=tokenizers,
            image_processor=image_processor,
            pipe=pipe,
        )
        self.tokenizer_max_length = kwargs.get("tokenizer_max_length", 4096)
        print(f"QwenImageProcessor tokenizer_max_length: {self.tokenizer_max_length}")
        self.prompt_template_encode = kwargs.get(
            "prompt_template_encode",
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        )
        self.prompt_template_encode_start_idx = kwargs.get(
            "prompt_template_encode_start_idx", 34
        )

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        processed_image, original_size = self._load_and_process_image(item)
        prompt = item["text"]

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(prompt)]

        txt_tokens = self.tokenizers[0](
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = (txt_tokens.input_ids[0],)
        attention_mask = (txt_tokens.attention_mask[0],)

        return {
            "pixel_values": processed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_size": torch.tensor(original_size),
        }


class QwenImageEditProcessor(BaseItemProcessor):
    def __init__(
        self,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        image_dir: str = None,
        pipe: Any = None,
        **kwargs,
    ):
        super().__init__(
            image_dir=image_dir,
            tokenizers=tokenizers,
            image_processor=image_processor,
            pipe=pipe,
        )

        self.tokenizer_max_length = kwargs.get("tokenizer_max_length", 1024)

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipe is None:
            raise ValueError(
                "QwenImageEditProcessor requires 'pipe' to be passed for text preprocessing."
            )

        processed_image, original_size = self._load_and_process_image(item)
        prompt = item["text"]

        cond_image = self._load_image_entry(item["condition_image"])
        cond_image = self.image_processor.resize(cond_image, original_size[0], original_size[1])
        cond_pixel_values = self.image_processor.preprocess(cond_image)
        template = self.pipe.prompt_template_encode
        txt = [template.format(prompt)]
        model_inputs = self.pipe.processor(
            text=txt,
            images=cond_image,
            padding=True,
            return_tensors="pt",
        )

        input_ids = (model_inputs.input_ids,)
        attention_mask = (model_inputs.attention_mask,)
        text_encoder_pixel_values = (model_inputs.pixel_values,)
        image_grid_thw = (model_inputs.image_grid_thw,)

        result = {
            "pixel_values": processed_image,
            "original_size": torch.tensor(original_size),
            "condition_pixel_values": cond_pixel_values,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_encoder_pixel_values": text_encoder_pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        return result


PROCESSOR_REGISTRY = {
    "flux-dev": FluxDevProcessor,
    "pixart-sigma": PixArtSigmaProcessor,
    "sd3": SD3Processor,
    "qwen-image": QwenImageProcessor,
    "qwen-image-edit": QwenImageEditProcessor,
}


class ImageTextDataset(Dataset):
    def __init__(
        self,
        model_type: str,
        image_dir: str,
        metadata_file: str,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        **kwargs,
    ):

        self.model_type = model_type

        self.metadata = []
        self.process_metadata(metadata_file)

        ProcessorClass = PROCESSOR_REGISTRY.get(model_type)
        if ProcessorClass is None:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.processor = ProcessorClass(
            image_dir=image_dir,
            tokenizers=tokenizers,
            image_processor=image_processor,
            **kwargs,
        )

        self._preload_hf_datasets()
    
    def _preload_hf_datasets(self):
        required_splits = set()
        
        keys_to_scan = ["file_name", "image", "condition_image"]
        
        print(f"[Dataset] Scanning {len(self.metadata)} items for HF URIs...")
        for item in self.metadata:
            for key in keys_to_scan:
                val = item.get(key)
                if isinstance(val, str) and val.startswith("hf://"):
                    try:
                        # hf://dataset_name/split/index/col
                        content = val[5:]
                        parts = content.split("/")
                        if len(parts) >= 4:
                            split = parts[-3]
                            dataset_name = "/".join(parts[:-3])
                            required_splits.add((dataset_name, split))
                    except:
                        pass
        
        for d_name, split in required_splits:
            self.processor.preload_hf_dataset(d_name, split)
        
        if required_splits:
            print(f"[Dataset] Preload complete. Cached {len(required_splits)} HF dataset splits.")

    def process_metadata(self, metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    self.metadata.append(item)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        try:
            return self.processor(item)
        except Exception as e:
            print(
                f"Error loading item at index {idx} ({item.get('file_name', 'unknown')}): {e}. Trying next item."
            )
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(batch):
    if not batch:
        return {}
    
    elem = batch[0]
    collated_batch = {}

    if "pixel_values" in elem:
        collated_batch["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)

    if "condition_pixel_values" in elem:
        cond_list = [item["condition_pixel_values"] for item in batch if "condition_pixel_values" in item]
        if len(cond_list) == len(batch):
            collated_batch["condition_pixel_values"] = torch.cat(cond_list, dim=0)

    if "original_size" in elem:
        collated_batch["original_size"] = torch.stack([item["original_size"] for item in batch])

    tuple_like_attrs = ['input_ids', 'attention_mask', 'text_encoder_pixel_values', 'image_grid_thw']

    for attr in tuple_like_attrs:
        if attr in elem and elem[attr] is not None:
            tuple_len = len(elem[attr])
            
            # [[item0_idx0, item1_idx0...], [item0_idx1, item1_idx1...]]
            batched_lists = [[] for _ in range(tuple_len)]
            
            for item in batch:
                item_tuple = item[attr]
                for i in range(tuple_len):
                    batched_lists[i].append(item_tuple[i])
            
            collated_batch[attr] = tuple(torch.stack(l) for l in batched_lists)
        else:
            collated_batch[attr] = None

    return collated_batch

if __name__ == "__main__":
    import sys
    import numpy as np
    from torch.utils.data import DataLoader
    from shiva_dit.utils import get_pipeline

    image_dir = None
    metadata_file = "../dataset/MagicBrush/val.jsonl"

    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        sys.exit(1)

    print(f"Testing dataset with metadata: {metadata_file}")
    print(f"Image directory: {image_dir}")

    print("Initializing Tokenizer and Processor...")
    model_type = "qwen-image-edit"
    pipe, model_attrs = get_pipeline(model_type)
    tokenizers = (pipe.tokenizer,)
    
    dataset = ImageTextDataset(
        model_type=model_type,
        image_dir=image_dir,
        metadata_file=metadata_file,
        tokenizers=tokenizers, 
        image_processor=pipe.image_processor
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )

    print("\n--- Starting Test Iteration ---")
    for i, batch in enumerate(dataloader):
        print(f"\n[Batch {i}] Keys:", batch.keys())

        input_ids = batch["input_ids"]
        if isinstance(input_ids, tuple):
            print(f"  Input IDs (Tuple) Shape: {[t.shape for t in input_ids]}")
        else:
            print(f"  Input IDs Shape: {input_ids.shape}")
        
        if batch["attention_mask"] is not None:
            masks = batch["attention_mask"]
            print(f"  Attention Mask Shape: {[m.shape for m in masks]}")

        pixel_values = batch["pixel_values"]
        print(f"  Target Image (pixel_values) Shape: {pixel_values.shape}") 

        if "condition_pixel_values" in batch:
            cond_pixels = batch["condition_pixel_values"]
            print(f"  Condition Image (condition_pixel_values) Shape: {cond_pixels.shape}")
        else:
            print("  WARNING: 'condition_pixel_values' missing in batch! Is 'condition_image' in your jsonl?")

        B = pixel_values.shape[0]
        if "condition_pixel_values" in batch:
            assert batch["condition_pixel_values"].shape[0] == B, "Condition image batch size mismatch"
            assert batch["condition_pixel_values"].shape[1:] == pixel_values.shape[1:], "Resolution mismatch between target and condition"
        
        print("✅ Batch check passed.")
        break