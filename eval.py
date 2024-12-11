import argparse
import numpy as np
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModelEvaluator:
    def __init__(self, base_model_name, trained_model_path, eval_dataset_path, eval_dataset_config, device='cuda'):
        self.base_model_name = base_model_name
        self.trained_model_path = trained_model_path
        self.eval_dataset_path = eval_dataset_path
        self.eval_dataset_config = eval_dataset_config
        self.device = device

        if not torch.cuda.is_available() and self.device.startswith('cuda'):
            logger.warning("CUDA device not available, using CPU instead.")
            self.device = 'cpu'

        logger.info(f"Using device: {self.device}")
        # Load models
        self.base_model, self.tokenizer = self.load_model(self.base_model_name)
        self.trained_model, _ = self.load_model(self.trained_model_path)

    def load_model(self, model_name):
        logger.info(f"Loading model '{model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"{model_name} loaded and moved to {self.device}.")
        return model, tokenizer

    def load_and_format_dataset(self,  max_length=2048):
        def formatting_prompts_func(examples):
            if "text" in examples:
                return {"text": [example + " " + self.tokenizer.eos_token for example in examples["text"]]}
            elif "sentence" in examples:
                return {"text": [example + " " + self.tokenizer.eos_token for example in examples["sentence"]]}
            else:
                raise ValueError("Expected 'text' or 'sentence' in dataset examples.")
        split = "validation"
        dataset = load_dataset(self.eval_dataset_path, self.eval_dataset_config, split=split,
                               trust_remote_code=True,
                               max_length=max_length)


        dataset = dataset.map(formatting_prompts_func, batched=True)
        logger.info(f"Dataset '{self.eval_dataset_path}' with config '{self.eval_dataset_config}' loaded and formatted.")
        # Print dataset statistics
        logger.info(f"Dataset Statistics - Number of examples in the [[{split}]] split set: {len(dataset)}")
        return dataset

    def tokenize_dataset(self, dataset, max_length=256):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding="longest",
                max_length=max_length
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logger.info(f"Dataset tokenized with max_length={max_length}.")
        return tokenized_dataset

    def compute_perplexity(self, model, dataset):
        logger.info("Starting evaluation on dataset...")
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
            do_eval=True,
            report_to="none"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        perplexity = np.exp(eval_loss)
        logger.info(f"Perplexity: {perplexity}")
        return perplexity

    def generate_sentences(self, model, prompt, max_length=250, num_return_sequences=5, temperature=0.7, top_k=50, top_p=0.95):
        logger.info("Generating sentences based on prompt...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        model.eval()
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for idx, text in enumerate(generated_texts):
            logger.info(f"Generated text {idx + 1}: {text}")
        return generated_texts


    def evaluate(self, prompt=None, max_length=250, num_return_sequences=5, temperature=0.7, top_k=50, top_p=0.95):
        # Load and tokenize evaluation dataset
        eval_dataset = self.load_and_format_dataset()
        eval_tokenized_dataset = self.tokenize_dataset(eval_dataset)

        # Compute perplexity for base model
        logger.info("Evaluating base model...")
        base_perplexity = self.compute_perplexity(self.base_model, eval_tokenized_dataset)

        # Compute perplexity for trained model
        logger.info("Evaluating the trained model...")
        trained_perplexity = self.compute_perplexity(self.trained_model, eval_tokenized_dataset)

        # Generate sentences based on prompt, if provided
        if prompt:
            logger.info(f"Generating text from prompt: '{prompt}'")
            logger.info("Base model generations:")
            base_generations = self.generate_sentences(
                self.base_model, prompt, max_length, num_return_sequences, temperature, top_k, top_p
            )
            logger.info("Trained model generations:")
            trained_generations = self.generate_sentences(
                self.trained_model, prompt, max_length, num_return_sequences, temperature, top_k, top_p
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True, help="The name of the base model to use for evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Path to the evaluation dataset.")
    parser.add_argument("--eval_dataset_config", type=str, required=False, default="default", help="Configuration for the evaluation dataset.")
    parser.add_argument("--prompt", type=str, required=False, default=None, help="Prompt for text generation.")
    # Hyperparameters for text generation
    parser.add_argument("--max_length", type=int, default=250, help="Maximum length for generated sequences.")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="The number of highest probability vocabulary tokens to keep for top-k filtering.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Cumulative probability for top-p (nucleus) sampling.")

    args = parser.parse_args()

    evaluator = LanguageModelEvaluator(
        base_model_name=args.base_model_name,
        trained_model_path=args.model_path,
        eval_dataset_path=args.eval_dataset_path,
        eval_dataset_config=args.eval_dataset_config
    )

    evaluator.evaluate(
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()
