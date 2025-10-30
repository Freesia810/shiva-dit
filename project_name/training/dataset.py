# 文件: project_name/training/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from typing import Callable, Optional

class ImageTextDataset(Dataset):
    """
    更新后的数据集，支持多 tokenizer 和元数据 (如分辨率)
    
    参数:
    - metadata_file (str): .jsonl 文件，每行 {"file_name": "...", "text": "..."}
    - image_dir (str): 图像文件夹
    - tokenizer (Callable): 主 tokenizer (例如 FLUX 的 CLIP Tokenizer)
    - image_processor (Callable): 图像处理器
    - tokenizer_2 (Callable, optional): 辅助 tokenizer (例如 FLUX 的 T5 Tokenizer)
    - model_type (str): 用于判断需要哪些特定数据 (e.g., 'flux-dev')
    """
    def __init__(self, 
                 metadata_file: str, 
                 image_dir: str, 
                 tokenizer: Callable, 
                 image_processor: Callable,
                 tokenizer_2: Optional[Callable] = None,
                 model_type: str = "default"):
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_processor = image_processor
        self.model_type = model_type
        
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
        text = item['text']
        return_dict = {}
        
        # 1. 加载和处理图像
        image_path = os.path.join(self.image_dir, item['file_name'])
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 存储元数据，用于 PixArt/FLUX 的 adaLN
            original_size = (image.height, image.width)
            # 假设 image_processor 是标准 diffusers 处理器
            processed_image = self.image_processor(image)

        except Exception as e:
            print(f"警告：加载或处理图像 {image_path} 时出错: {e}")
            # 尝试加载下一个
            return self.__getitem__((idx + 1) % len(self))
            
        # 2. 处理文本 (Tokenizer 1 - 主)
        tokenized_1 = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            # 确保 tokenizer 有 model_max_length 属性
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        # 适配 FLUX (它需要 input_ids_1 和 input_ids_2)
        key_1 = "input_ids_1" if self.model_type == "flux-dev" else "input_ids"
        return_dict[key_1] = tokenized_1['input_ids'].squeeze(0)

        # 3. 处理文本 (Tokenizer 2 - 辅助, e.g., for FLUX T5)
        if self.tokenizer_2 is not None:
            tokenized_2 = self.tokenizer_2(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt"
            )
            return_dict["input_ids_2"] = tokenized_2['input_ids'].squeeze(0)
        
        # 4. 组合返回
        return_dict["pixel_values"] = processed_image.squeeze(0) 
        
        # (关键) 添加 DiT 模型需要的额外条件
        return_dict["original_size"] = torch.tensor(original_size)
        # 假设从 (0,0) 裁切
        return_dict["crop_coords_top_left"] = torch.tensor([0, 0])

        return return_dict

def collate_fn(batch):
    """
    自定义 collate_fn 来处理不同 key 的批处理
    并动态创建 'added_cond_kwargs'
    """
    # 动态收集所有key
    elem = batch[0]
    keys = elem.keys()
    collated_batch = {}

    for key in keys:
        try:
            collated_batch[key] = torch.stack([d[key] for d in batch])
        except Exception:
            # (处理 attention_mask 等可能不存在的 key)
            pass

    # --- (关键) 为 PixArt/FLUX 准备 'added_cond_kwargs' ---
    if "original_size" in collated_batch and "crop_coords_top_left" in collated_batch:
        
        collated_batch["added_cond_kwargs"] = {
            "resolution": collated_batch.get("original_size"),
            "crop_coords": collated_batch.get("crop_coords_top_left"),
            # (如果数据集提供了 aspect_ratio，也在这里添加)
            # "aspect_ratio": collated_batch.get("aspect_ratio") 
        }

    return collated_batch