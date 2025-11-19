import os
from abc import ABC, abstractmethod

class AbstractLogger(ABC):
    @abstractmethod
    def log_scalars(self, data_dict, step, prefix=""):
        pass

    @abstractmethod
    def log_image(self, tag, image, step, prefix=""):
        pass

    @abstractmethod
    def add_scalars(self, main_tag, tag_scalar_dict, step, prefix=""):
        pass

    @abstractmethod
    def close(self):
        pass

class TensorBoardLogger(AbstractLogger):
    def __init__(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            import torchvision.transforms as T
        except ImportError:
            print("TensorBoardLogger Error: Install tensorboard and torchvision (pip install tensorboard torchvision)")
            raise

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.pil_to_tensor = T.ToTensor()
        print(f"TensorBoard logging enabled at {log_dir}")

    def log_scalars(self, data_dict, step, prefix=""):
        for k, v in data_dict.items():
            self.writer.add_scalar(f"{prefix}{k}", v, step)

    def log_image(self, tag, image, step, prefix=""):
        try:
            import torch
            if isinstance(image, torch.Tensor):
                self.writer.add_image(f"{prefix}{tag}", image, step)
            else:
                image_tensor = self.pil_to_tensor(image)
                self.writer.add_image(f"{prefix}{tag}", image_tensor, step)
        except Exception as e:
            print(f"TensorBoard log_image Error: {e}")

    def add_scalars(self, main_tag, tag_scalar_dict, step, prefix=""):
        self.writer.add_scalars(f"{prefix}{main_tag}", tag_scalar_dict, step)

    def close(self):
        self.writer.close()


class WandbLogger(AbstractLogger):
    def __init__(self, project, name, config_dict=None, entity=None):
        try:
            import wandb
        except ImportError:
            print("WandbLogger Error: Install wandb (pip install wandb)")
            raise

        self.wandb = wandb
        wandb.init(project=project, name=name, entity=entity, config=config_dict)
        print("Weights & Biases logging enabled.")

    def log_scalars(self, data_dict, step, prefix=""):
        prefixed_data = {f"{prefix}{k}": v for k, v in data_dict.items()}
        self.wandb.log(prefixed_data, step=step)

    def log_image(self, tag, image, step, prefix=""):
        self.wandb.log({f"{prefix}{tag}": self.wandb.Image(image)}, step=step)

    def add_scalars(self, main_tag, tag_scalar_dict, step, prefix=""):
        chart_name = f"{prefix}{main_tag}"
        log_data = { chart_name: tag_scalar_dict }
        self.wandb.log(log_data, step=step)

    def close(self):
        self.wandb.finish()

class MLflowLogger(AbstractLogger):
    def __init__(self, experiment_name: str, run_name: str = None, config_dict: dict = None, tracking_uri: str = None):
        try:
            import mlflow
        except ImportError:
            print("MLflowLogger Error: Install mlflow (pip install mlflow)")
            raise
        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.run = self.mlflow.start_run(run_name=run_name)
        print(f"MLflow logging enabled. Run ID: {self.run.info.run_id}")
        if config_dict:
            flat_config = self._flatten_dict(config_dict)
            for i in range(0, len(flat_config), 100):
                batch = dict(list(flat_config.items())[i:i+100])
                self.mlflow.log_params(batch)

    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, (list, tuple)):
                items[new_key] = str(v)
            else:
                items[new_key] = v
        return items

    def log_scalars(self, data_dict, step, prefix=""):
        for k, v in data_dict.items():
            self.mlflow.log_metric(f"{prefix}{k}", v, step=step)

    def log_image(self, tag, image, step, prefix=""):
        try:
            full_tag_path = f"{prefix}{tag}"
            filename = f"step_{step:06d}.png"
            self.mlflow.log_image(image, artifact_file=os.path.join(full_tag_path, filename))
        except Exception as e:
            print(f"MLflow log_image Error: {e}")

    def add_scalars(self, main_tag, tag_scalar_dict, step, prefix=""):
        for k, v in tag_scalar_dict.items():
            self.mlflow.log_metric(f"{prefix}{main_tag}/{k}", v, step=step)

    def close(self):
        self.mlflow.end_run()
        print("MLflow run finished.")


class NoOpLogger(AbstractLogger):

    def log_scalars(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class LoggerCollection(AbstractLogger):
    def __init__(self, loggers: list):
        self.loggers = loggers

    def log_scalars(self, data_dict, step, prefix=""):
        for logger in self.loggers:
            logger.log_scalars(data_dict, step, prefix)

    def log_image(self, tag, image, step, prefix=""):
        for logger in self.loggers:
            logger.log_image(tag, image, step, prefix)

    def add_scalars(self, main_tag, tag_scalar_dict, step, prefix=""):
        for logger in self.loggers:
            logger.add_scalars(main_tag, tag_scalar_dict, step, prefix)

    def close(self):
        for logger in self.loggers:
            logger.close()
