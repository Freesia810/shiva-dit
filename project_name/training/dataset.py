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
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        prompt = item["text"]
        # TODO: negative prompt support
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
    for item in batch:
        pixel_values_list.append(item['pixel_values'])
        original_size_list.append(item['original_size'])
        
        for i in range(input_ids_tuple_len):
            input_ids_lists[i].append(item['input_ids'][i].squeeze(0))

            
    stacked_pixel_values = torch.stack(pixel_values_list)
    stacked_original_size = torch.stack(original_size_list)
    stacked_input_ids_tuple = tuple(
        torch.stack(id_list) for id_list in input_ids_lists
    )
    
    collated_batch = {
        "pixel_values": stacked_pixel_values,
        "input_ids": stacked_input_ids_tuple, # 格式为 (tensor_1, tensor_2, ...)
        "original_size": stacked_original_size
    }

    return collated_batch
