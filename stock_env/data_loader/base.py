from abc import abstractmethod
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl

class BaseDataLoader:
    
    def save(self, path):
        save_to_pkl(path, self)
    
    @classmethod
    def load(cls, path):
        return load_from_pkl(path)
    
    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError


class BaseTaskLoader:
    
    @abstractmethod
    def reset_task(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def sample_task(self, *args, **kwargs):
        raise NotImplementedError