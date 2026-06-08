from trl import SFTTrainer, SFTConfig

class Trainer:
    """
    Responsible for fine-tuning the model on medical dataset.
    """
    
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def train(self, train_data, test_data):
        """Fine-tunes model on medical dataset"""
        sft_config = SFTConfig(
            output_dir="outputs/model",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            dataset_text_field="text",
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=test_data,
            args=sft_config,
        )
        
        print("Starting training...")
        trainer.train()
        print("Training complete! ✅")
        return trainer
