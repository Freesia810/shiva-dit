import torch
from shiva_dit.training.logging import LoggerCollection

class MetricManager:
    def __init__(self, logger: LoggerCollection, accelerator, grouping_rules: dict = None):
        """
        Initialize the MetricManager.

        Args:
            logger: LoggerCollection Instance
            accelerator: Accelerator Instance
            grouping_rules (dict): 
                Example: 
                {
                    # "shiva/ratio/avg_vs_target" Card Name
                    "shiva/ratio/avg_vs_target": [
                        "shiva/ratio/avg_ratio",  # Line1
                        "shiva/ratio/target_ratio"  # Line2
                    ]
                }
        """
        self.logger = logger
        self.accelerator = accelerator
        self.grouping_rules = grouping_rules if grouping_rules is not None else {}
        self.metrics_cache = {}

    def update(self, metrics_dict: dict):
        try:
            gathered_metrics = self.accelerator.gather_for_metrics(metrics_dict)
        except Exception as e:
            if self.accelerator.is_main_process:
                print(
                    f"Warning: Metric gathering failed (Error: {e}). Logging main process metrics only."
                )
            gathered_metrics = metrics_dict

        if not self.accelerator.is_main_process:
            return

        for k, v in gathered_metrics.items():
            if isinstance(v, torch.Tensor):
                self.metrics_cache[k] = v.detach().item()
            elif isinstance(v, (float, int)):
                self.metrics_cache[k] = v

    def log_step(self, global_step: int):
        if not self.accelerator.is_main_process or not self.metrics_cache:
            return

        scalars_to_log_individually = self.metrics_cache.copy()
        processed_keys = set()

        for group_name, keys_to_group in self.grouping_rules.items():
            group_dict = {}
            for key in keys_to_group:
                if key in scalars_to_log_individually:
                    # 'shiva/ratio/avg_ratio' -> 'avg_ratio'
                    simple_name = key.split('/')[-1]
                    group_dict[simple_name] = scalars_to_log_individually[key]
                    processed_keys.add(key)

            if group_dict:
                self.logger.add_scalars(group_name, group_dict, global_step)

        individual_metrics_to_log = {}
        for key, value in scalars_to_log_individually.items():
            if key not in processed_keys:
                individual_metrics_to_log[key] = value

        if individual_metrics_to_log:
            self.logger.log_scalars(individual_metrics_to_log, step=global_step)

        self.metrics_cache.clear()

    def get_metric(self, key: str, default=0.0):
        return self.metrics_cache.get(key, default)
