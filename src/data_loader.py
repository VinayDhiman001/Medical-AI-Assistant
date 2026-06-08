from datasets import load_dataset
from typing import Tuple

class DataLoader:
    """
    Responsible for loading and splitting the medical dataset.
    Single responsibility - only handles data loading.
    """
    
    def __init__(self, config: dict):
        # Store data config
        self.config = config
        # Dataset stored here after loading
        self.dataset = None
    
    def load(self):
        """Loads dataset from HuggingFace"""
        print(f"Loading dataset: {self.config['dataset_name']}")
        self.dataset = load_dataset(self.config['dataset_name'])
        print(f"Dataset loaded! Total samples: {len(self.dataset['train'])}")
        return self
    
    def split(self) -> Tuple:
        """Splits dataset into train and test"""
        split = self.dataset['train'].train_test_split(
            test_size=self.config['test_split'],
            seed=42
        )
        train_data = split['train']
        test_data = split['test']
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        return train_data, test_data
    
    def get_sample(self, index: int) -> dict:
        """Returns a single sample from dataset"""
        return self.dataset['train'][index]
