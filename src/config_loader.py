import yaml

class ConfigLoader:
    """
    Responsible for loading and accessing config values.
    Single place to manage all project settings.
    """

    def __init__(self, config_path: str):
        # Store the path
        self.config_path = config_path
        # Load config when object is created
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Reads yaml file and returns as dictionary"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self) -> dict:
        """Returns model settings"""
        return self.config['model']

    def get_training_config(self) -> dict:
        """Returns training settings"""
        return self.config['training']

    def get_data_config(self) -> dict:
        """Returns data settings"""
        return self.config['data']
