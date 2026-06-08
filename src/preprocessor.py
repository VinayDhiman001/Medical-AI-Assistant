class Preprocessor:
    """
    Responsible for cleaning and formatting dataset.
    Converts raw data into format suitable for fine-tuning.
    """
    
    def __init__(self, config: dict):
        # Store model config
        self.config = config
    
    def format_prompt(self, sample: dict) -> str:
        """
        Formats raw sample into Mistral instruction format.
        This is the format Mistral 7B was trained to understand.
        """
        prompt = f"""<s>[INST] {sample['input']}
{sample['instruction']} [/INST]
{sample['output']} </s>"""
        return prompt
    
    def clean(self, dataset):
        """Removes missing values and duplicates"""
        dataset = dataset.filter(
            lambda x: x['input'] != '' and x['output'] != ''
        )
        print(f"After cleaning: {len(dataset)} samples")
        return dataset
    
    def preprocess(self, dataset):
        """Applies formatting to entire dataset"""
        dataset = self.clean(dataset)
        dataset = dataset.map(
            lambda x: {'text': self.format_prompt(x)}
        )
        print("Formatting done! ✅")
        return dataset
