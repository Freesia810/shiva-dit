from typing import Callable


class UNetScheduler:
    def __init__(self, timesteps: int, dst_recompute_timesteps: list, attn_recompute_timesteps: list):
        assert set(dst_recompute_timesteps).issubset(set(attn_recompute_timesteps)), "dst_recompute_timesteps should be subset of attn_recompute_timesteps"

        self.dst_recompute_t = dst_recompute_timesteps
        self.attn_recompute_t = attn_recompute_timesteps

        self.unet_structure = self._initialize_structure()
        self.blockidx2name = {}
        accumulated_sum = 0
        # Assigning block name for every index in its range
        for block_name, block_count in self.unet_structure.items():
            for idx in range(accumulated_sum, accumulated_sum + block_count):
                self.blockidx2name[idx] = block_name
            accumulated_sum += block_count

        self.timesteps = timesteps
        self.unet_block_count = sum(self.unet_structure.values())
        self.total_block_counts = self.unet_block_count * timesteps
        self.block_counter = -1
        self.block_storage = self._initialize_storage(self.unet_structure)
        self.current_timestep = -1
        self.current_block_idx = -1

        self.first_block_idx = self._get_first_block_of_each_type()

    def _get_first_block_of_each_type(self):
        """
        Precompute and return a set of indices representing the first occurrence of each block type.
        """
        first_block_indices = set()
        seen_blocks = set()

        for idx, block_name in self.blockidx2name.items():
            if block_name not in seen_blocks:
                first_block_indices.add(idx)
                seen_blocks.add(block_name)

        return first_block_indices
    
    def _initialize_storage(self, unet_structure: dict) -> dict:
        storage = {}
        for block_name, _ in unet_structure.items():
            storage[block_name] = {
                "dst_idx": None,
                "A": None,
                "A_inv": None,
            }
        return storage

    def _initialize_structure(self) -> dict:
        structure = {
            "CrossAttnDownBlock2D-1": ((64, 64), 2, 2),
            "CrossAttnDownBlock2D-2": ((32, 32), 10, 2),
            "MidBlock2DCrossAttn": ((32, 32), 10, 1),
            "CrossAttnUpBlock2D-1_2": ((32, 32), 10, 3),
            "CrossAttnUpBlock2D-2_2": ((64, 64), 2, 3),
        }

        # Multiply the number of blocks by the repeat count for each block type
        structure = {
            block_name: num_blocks * repeat_count
            for block_name, (feature_map_size, num_blocks, repeat_count) in structure.items()
        }
        return structure

    def step(self):
        if self.block_counter >= self.total_block_counts:
            print("All blocks and repetitions have been processed.")
            self.reset()
            return False

        self.block_counter += 1
        self.current_timestep = self.block_counter // self.unet_block_count
        self.current_block_idx = self.block_counter % self.unet_block_count

        if_recompute_t = self.current_timestep in self.attn_recompute_t
        if_recompute_block = self.current_block_idx in self.first_block_idx
        if_recompute_attn = if_recompute_t and if_recompute_block

        return if_recompute_attn

    def get_current_t_block(self):
        current_block_name = self.blockidx2name.get(self.current_block_idx)
        if_first_block = self.current_block_idx in self.first_block_idx
        return self.current_timestep, current_block_name, if_first_block

    def get_dst_idx(self, compute_fn: Callable, *args):
        current_timestep, current_block, if_first_block = self.get_current_t_block()

        if current_timestep in self.dst_recompute_t and if_first_block:
            self.block_storage[current_block]["dst_idx"] = compute_fn(*args)
        return self.block_storage[current_block]["dst_idx"]

    def get_A(self, compute_fn: Callable, *args):
        current_timestep, current_block, if_first_block = self.get_current_t_block()
        if current_timestep in self.attn_recompute_t and if_first_block:
            self.block_storage[current_block]["A"], self.block_storage[current_block]["A_inv"] = compute_fn(*args)
        return self.block_storage[current_block]["A"], self.block_storage[current_block]["A_inv"]

    def reset(self):
        self.block_counter = 0
        self.block_storage = self._initialize_storage(self.unet_structure)
