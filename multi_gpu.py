import argparse
import numpy as np
import torch
import logging
import os
import torch.distributed as dist

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    # If using torchrun, these environment variables are set automatically
    # Initialize the default distributed process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

class LanguageModelTrainer:
    def __init__(self, model_name, train_dataset_path, eval_dataset_path, output_dir,
                 target_lang,
                 max_length=256):
        self.lora_alpha = None
        self.r = None
        self.target_lang = target_lang
        self.model_name = model_name
        self.train_dataset_path = train_dataset_path
        self.eval_dataset_path = eval_dataset_path
        self.output_dir = output_dir
        self.max_length = max_length

        # Local rank derived from environment variables set by torchrun
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set the current device
        torch.cuda.set_device(self.local_rank)

        logger.info(f"Local Rank: {self.local_rank}. Initializing model/tokenizer ...")
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        logger.info(f"Loading model '{self.model_name}'...")
        # Do not move model to device here; Trainer will handle device placement.
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"{self.model_name} loaded.")
        return model, tokenizer

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def load_and_format_dataset(self, dataset_name, dataset_config, split, limit=None, max_length=None):
        def formatting_prompts_func(examples):
            if "text" in examples:
                return {"text": [example + " " + self.tokenizer.eos_token for example in examples["text"]]}
            elif "sentence" in examples:
                return {"text": [example + " " + self.tokenizer.eos_token for example in examples["sentence"]]}
            else:
                raise ValueError("Expected 'text' or 'sentence' in dataset examples.")

        logger.info(f"Loading dataset '{dataset_name}' for language '{self.target_lang}'...")
        dataset = load_dataset(
            dataset_name, self.target_lang,
            split=split,
            trust_remote_code=True,
            streaming=False,
        )
        dataset = dataset.filter(lambda example: example.get("text") is not None and len(example["text"].strip()) > 0, num_procs=32)

        # Limit dataset size for debugging
        if limit and split == "train":
            dataset = dataset.take(limit)
            logger.info(f"Dataset size after limiting: {len(list(dataset))}")

        dataset = dataset.map(formatting_prompts_func, batched=True, num_procs=32)
        logger.info(f"Dataset loaded and formatted for language '{self.target_lang}'.")
        return dataset

    def tokenize_dataset(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, num_procs=32)
        logger.info(f"Dataset tokenized with max_length={self.max_length}.")
        return tokenized_dataset

    def apply_lora(self, model, r, lora_alpha):
        # Apply LoRA
        self.r = r
        self.lora_alpha = lora_alpha
        logger.info(f"Applying LoRA to the model r={self.r} ; alpha={self.lora_alpha}...")
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules='all-linear',
            fan_in_fan_out=False
        )
        lora_model = get_peft_model(model, lora_config)
        logger.info("LoRA applied successfully. Parameter count of the LoRA model:")
        self.print_trainable_parameters(lora_model)
        return lora_model

    def train_model(self, model, dataset, train_type, train_batch_size, max_steps, gradient_accumulation_steps):
        # Construct output directory name
        output_dir = f"{self.output_dir}/{self.model_name.replace('/', '-')}_{train_type}_{self.target_lang}_{self.train_dataset_path.replace('/', '-')}_max_steps-{max_steps}"
        if train_type == "adapter":
            output_dir += f"_r_{self.r}_alpha_{self.lora_alpha}"

        logger.info(f"Starting training: {train_type}... Output directory: {output_dir}")
        logger.info(f"Training for {max_steps} steps...")

        # Trainer should handle local rank automatically.
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=2500,
            save_total_limit=1,
            logging_dir='./logs',
            logging_steps=1000,
            report_to="none",
            max_steps=max_steps,
            learning_rate=2e-5,
            fp16=True,  # Enable mixed-precision training
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,  # Often needed for large models
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        dataset = dataset.with_format("torch")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()
        logger.info(f"Training ({train_type}) completed.")

    def compute_perplexity(self, model, dataset):
        logger.info("Starting evaluation on dataset...")
        model.eval()
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            logging_steps=100,
            do_eval=True,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
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
        logger.info(f"eval_loss: {eval_loss}")
        logger.info(f"Perplexity: {perplexity}")
        return perplexity

    def generate_sentences(self, model, prompt, max_length=200, num_return_sequences=3,
                           temperature=0.4, top_k=50, top_p=0.95, experiment_name="experiment"):
        logger.info("Generating sentences based on prompt...")
        model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
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
        filename = f"generated_{experiment_name}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"Prompt: {prompt}\n\n")
            for idx, text in enumerate(generated_texts):
                file.write(f"Generated text {idx + 1}:\n{text}\n\n")
        logger.info(f"Generated texts saved to {filename}")


def main():
    setup_distributed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the model to use for training or evaluation.")
    parser.add_argument("--train_dataset_path", type=str, required=False, help="Path to the training dataset.")
    parser.add_argument("--train_dataset_config", type=str, required=False,
                        help="Configuration for the training dataset.")
    parser.add_argument("--target_lang", type=str, default="fao", help="Target language for the dataset.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--max_steps", type=int, default=20000, help="Max training steps.")
    parser.add_argument("--eval_dataset_path", type=str, default=None, required=False,
                        help="Path to the evaluation dataset.")
    parser.add_argument("--eval_dataset_config", type=str, default=None, required=False,
                        help="Configuration for the evaluation dataset.")
    parser.add_argument("--train_type", type=str, choices=["adapter", "fine-tune"], required=False,
                        help="Training type: 'adapter' for LoRA or 'fine-tune' for full model.")
    parser.add_argument("--debug_limit", type=int, default=None,
                        help="Limit number of training examples for debugging.")
    parser.add_argument("--r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="Directory to save trained models.")
    parser.add_argument("--do_eval", action='store_true', help="Only evaluate the model on the evaluation dataset.")

    args = parser.parse_args()

    trainer = LanguageModelTrainer(
        model_name=args.model_name,
        target_lang=args.target_lang,
        max_length=args.max_length,
        train_dataset_path=args.train_dataset_path,
        eval_dataset_path=args.eval_dataset_path,
        output_dir=args.output_dir
    )

    if args.do_eval:
        assert args.eval_dataset_path
        eval_dataset = trainer.load_and_format_dataset(args.eval_dataset_path,
                                                       args.eval_dataset_config,
                                                       split="validation", max_length=args.max_length)
        eval_tokenized_dataset = trainer.tokenize_dataset(eval_dataset)
        trainer.compute_perplexity(trainer.model, eval_tokenized_dataset)
    else:
        assert args.train_dataset_path
        assert args.train_type
        train_dataset = trainer.load_and_format_dataset(args.train_dataset_path,
                                                        args.train_dataset_config,
                                                        limit=args.debug_limit, split="train",
                                                        max_length=args.max_length)
        train_tokenized_dataset = trainer.tokenize_dataset(train_dataset)
        train_tokenized_dataset = train_tokenized_dataset.with_format("torch")

        if args.train_type == "adapter":
            model = trainer.apply_lora(trainer.model, args.r, args.lora_alpha)
            trainer.train_model(model, train_tokenized_dataset,
                                train_type="adapter",
                                train_batch_size=args.train_batch_size,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                max_steps=args.max_steps)
        elif args.train_type == "fine-tune":
            trainer.train_model(trainer.model, train_tokenized_dataset,
                                train_type="fine-tune",
                                train_batch_size=args.train_batch_size,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                max_steps=args.max_steps)

        if args.eval_dataset_path:
            logger.info("Evaluating the trained model...")
            eval_dataset = trainer.load_and_format_dataset(args.eval_dataset_path,
                                                           args.eval_dataset_config,
                                                           split="validation", max_length=args.max_length)
            eval_dataset = eval_dataset.take(5000)
            eval_tokenized_dataset = trainer.tokenize_dataset(eval_dataset)
            eval_tokenized_dataset = eval_tokenized_dataset.with_format("torch")

            trainer.compute_perplexity(trainer.model, eval_tokenized_dataset)

if __name__ == "__main__":
    main()
