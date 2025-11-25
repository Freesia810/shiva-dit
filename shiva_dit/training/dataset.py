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
        image_dir: str,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        pipe: Any = None,
        **kwargs,
    ):
        self.image_dir = image_dir
        self.tokenizers = tokenizers
        self.image_processor = image_processor
        self.pipe = pipe
        self.hf_datasets_cache = {}

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
                        print(
                            f"[Dataset] Loading HF dataset split into memory: {dataset_name} [{split}]..."
                        )
                        # Loading dataset with keep_in_memory=False by default (uses disk cache / mmap)
                        self.hf_datasets_cache[cache_key] = load_dataset(
                            dataset_name, split=split
                        )

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
        image_dir: str,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        pipe: Any = None,
        **kwargs,
    ):
        super().__init__(
            image_dir,
            tokenizers,
            image_processor,
            pipe,
        )
        self.tokenizer_max_length = kwargs.get("tokenizer_max_length", 1024)
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
        image_dir: str,
        tokenizers: Tuple[Callable],
        image_processor: VaeImageProcessor,
        pipe: Any = None,
        **kwargs,
    ):
        super().__init__(
            image_dir,
            tokenizers,
            image_processor,
            pipe,
        )

        self.tokenizer_max_length = kwargs.get("tokenizer_max_length", 1024)

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        processed_image, original_size = self._load_and_process_image(item)

        prompt = item["text"]
        txt_tokens = self.tokenizers[0](
            [prompt],
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "pixel_values": processed_image,
            "input_ids": (txt_tokens.input_ids[0],),
            "attention_mask": (txt_tokens.attention_mask[0],),
            "original_size": torch.tensor(original_size),
            "raw_text": prompt,
        }

        if "condition_image" in item:
            try:
                # robust loader
                cond_image = self._load_image_entry(item["condition_image"])
                cond_pixel_values = self.image_processor.preprocess(cond_image)
                result["condition_pixel_values"] = cond_pixel_values
            except Exception as e:
                print(f"Warning: Failed to load condition image: {e}")

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
    elem = batch[0]
    collated_batch = {}

    pixel_values_list = []
    original_size_list = []

    input_ids_tuple_len = len(elem["input_ids"])
    input_ids_lists = [[] for _ in range(input_ids_tuple_len)]

    has_attention_mask = "attention_mask" in elem
    attention_mask_lists = (
        [[] for _ in range(input_ids_tuple_len)] if has_attention_mask else None
    )

    for item in batch:
        pixel_values_list.append(item["pixel_values"])
        original_size_list.append(item["original_size"])

        for i in range(input_ids_tuple_len):
            input_ids_lists[i].append(item["input_ids"][i])

        if has_attention_mask:
            for i in range(input_ids_tuple_len):
                attention_mask_lists[i].append(item["attention_mask"][i])

    collated_batch["pixel_values"] = torch.concat(pixel_values_list, dim=0)
    collated_batch["original_size"] = torch.stack(original_size_list)
    collated_batch["input_ids"] = tuple(
        torch.stack(id_list) for id_list in input_ids_lists
    )

    if has_attention_mask:
        collated_batch["attention_mask"] = tuple(
            torch.stack(mask_list) for mask_list in attention_mask_lists
        )
    else:
        collated_batch["attention_mask"] = None

    if "condition_pixel_values" in elem:
        cond_list = [
            item["condition_pixel_values"]
            for item in batch
            if "condition_pixel_values" in item
        ]
        if len(cond_list) == len(batch):
            collated_batch["condition_pixel_values"] = torch.concat(cond_list, dim=0)

    if "raw_text" in elem:
        collated_batch["raw_text"] = [item["raw_text"] for item in batch]

    return collated_batch
