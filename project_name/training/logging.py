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
        log_data = {f"{prefix}{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self.wandb.log(log_data, step=step)

    def close(self):
        self.wandb.finish()


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
