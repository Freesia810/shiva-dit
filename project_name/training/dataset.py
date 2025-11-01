from typing import Callable, Tuple
import torch
from torch.utils.data import Dataset
import os
import json
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image

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
        self.image_dir = image_dir
        self.tokenizers = tokenizers
        self.image_processor = image_processor

        self.metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'file_name' in item and 'text' in item:
                    self.metadata.append(item)

        if self.model_type == "qwen-image":
            self.tokenizer_max_length = kwargs.get("tokenizer_max_length", 1024)
            self.prompt_template_encode = kwargs.get(
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                "{}",
            )
            self.prompt_template_encode_start_idx = kwargs.get("prompt_template_encode_start_idx", 34)

        self.pipe = kwargs.get("pipe", None)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        return_dict = {}

        # preprocess image
        image_path = os.path.join(self.image_dir, item['file_name'])
        try:
            image = load_image(image_path)
            original_size = (image.height, image.width)
            processed_image = self.image_processor.preprocess(image)
            if self.model_type == "pixart-sigma":
                from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
                    ASPECT_RATIO_1024_BIN,
                )
                height, width = self.image_processor.classify_height_width_bin(
                    original_size[0], original_size[1], ASPECT_RATIO_1024_BIN
                )
                processed_image = self.image_processor.resize_and_crop_tensor(
                    processed_image, height, width
                )

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        prompt = item["text"]
        if self.model_type == "flux-dev":
            # (clip, t5)
            input_ids = tuple(
                tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length or 512,
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt",
                ).input_ids
                for tokenizer in self.tokenizers
            )
        elif self.model_type == "pixart-sigma":
            prompt = self.pipe._text_preprocessing(prompt, clean_caption=True)
            max_sequence_length = 300
            text_inputs = self.tokenizers[0](
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = (text_inputs.input_ids,)
            attention_mask = (text_inputs.attention_mask,)
            return_dict["attention_mask"] = attention_mask
        elif self.model_type == "sd3":
            # (clip, clip, t5)
            assert len(self.tokenizers) == 3, "SD3 model_type requires 3 tokenizers"
            clip_tokenizer_max_length = self.tokenizers[0].model_max_length or 77
            t5_tokenizer_max_length = 256

            input_ids = [
                tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=clip_tokenizer_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
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
                ).input_ids
            )
            input_ids = tuple(input_ids)
        elif self.model_type == "qwen-image":
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(prompt)]
            txt_tokens = self.tokenizer(
                txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            input_ids = (txt_tokens.input_ids,)
            attention_mask = (txt_tokens.attention_mask,)
            return_dict["attention_mask"] = attention_mask
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return_dict["pixel_values"] = processed_image
        return_dict["input_ids"] = input_ids
        return_dict["original_size"] = torch.tensor(original_size)

        return return_dict

def collate_fn(batch):
    elem = batch[0]

    pixel_values_list = []
    original_size_list = []

    input_ids_tuple_len = len(elem['input_ids'])
    input_ids_lists = [[] for _ in range(input_ids_tuple_len)]

    if 'attention_mask' in elem:
        attention_mask_lists = [[] for _ in range(input_ids_tuple_len)]
        for item in batch:
            for i in range(input_ids_tuple_len):
                attention_mask_lists[i].append(item['attention_mask'][i].squeeze(0))
        stacked_attention_mask_tuple = tuple(
            torch.stack(mask_list) for mask_list in attention_mask_lists
        )

    for item in batch:
        pixel_values_list.append(item['pixel_values'])
        original_size_list.append(item['original_size'])

        for i in range(input_ids_tuple_len):
            input_ids_lists[i].append(item['input_ids'][i].squeeze(0))

    stacked_pixel_values = torch.concat(pixel_values_list) # we can't increase dim
    stacked_original_size = torch.stack(original_size_list)
    stacked_input_ids_tuple = tuple(
        torch.stack(id_list) for id_list in input_ids_lists
    )

    collated_batch = {
        "pixel_values": stacked_pixel_values,
        "input_ids": stacked_input_ids_tuple,
        "original_size": stacked_original_size,
        "attention_mask": stacked_attention_mask_tuple if 'attention_mask' in elem else None,
    }

    return collated_batch
