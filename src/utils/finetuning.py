"""
Fine-tuning utilities for Mistral-7B-Instruct-v0.2 model.
Prepares training data and provides fine-tuning scripts.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    instruction: str
    input: str
    output: str
    
    def to_mistral_format(self) -> Dict:
        """Convert to Mistral chat format."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Operations Research modeler. Provide mathematically correct and properly formatted optimization formulations."
                },
                {
                    "role": "user",
                    "content": f"{self.instruction}\n\nContext:\n{self.input}"
                },
                {
                    "role": "assistant",
                    "content": self.output
                }
            ]
        }


class FineTuningDataPreparer:
    """Prepare training data for fine-tuning."""
    
    def __init__(self, output_dir: Path = Path("finetuning_data")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_correct_formulation_examples(self) -> List[TrainingExample]:
        """
        Create examples of correct formulations for common OR problems.
        These examples will help the model learn correct modeling patterns.
        """
        examples = []
        
        examples.append(TrainingExample(
            instruction="Formulate a production planning problem with inventory",
            input="""Acme Manufacturing Company has a contract to deliver home windows over the next 6 months.
The successive demands for the six periods are: 100, 250, 190, 140, 220, and 110 units.
Production cost per window varies monthly: $50, $45, $55, $48, $52, and $50 for months 1-6 respectively.
Acme can produce more than needed and hold excess units. Storage costs are $8 per window per month, assessed on end-of-month inventory.
Develop an LP to determine an optimum production schedule.""",
            output="""**Problem Analysis:**
This is a production planning problem where Acme needs to meet demand over 6 months while minimizing total production and storage costs. The company can use inventory to smooth production across months.

**Decision Variables (for months t = 1, ..., 6):**
- p_t = number of windows produced in month t (nonnegative)
- I_t = end-of-month inventory of windows in month t (nonnegative)

**Given Data:**
- Demands: d_1 = 100, d_2 = 250, d_3 = 190, d_4 = 140, d_5 = 220, d_6 = 110 (units)
- Production costs: c_1^(p) = 50, c_2^(p) = 45, c_3^(p) = 55, c_4^(p) = 48, c_5^(p) = 52, c_6^(p) = 50 (dollars per window)
- Holding cost: h = 8 dollars per window per month
- Initial inventory: I_0 = 0

**Objective (minimize total production + holding cost):**
min Z = Σ_{t=1}^6 c_t^(p) p_t + 8 Σ_{t=1}^6 I_t

**Mathematical Formulation:**

$$
\\begin{aligned}
\\text{Minimize} \\quad & \\sum_{t=1}^{6} c_t^{(p)} p_t + 8 \\sum_{t=1}^{6} I_t \\\\
\\text{Subject to} \\quad & I_0 + p_1 - d_1 = I_1 \\quad \\text{(with } I_0 = 0\\text{)} \\\\
& I_{t-1} + p_t - d_t = I_t, \\quad \\forall t = 2, \\ldots, 6 \\\\
& p_t \\ge 0, \\quad I_t \\ge 0, \\quad \\forall t = 1, \\ldots, 6
\\end{aligned}
$$

**Explanation:**
The objective minimizes total production costs (which vary by month) plus total holding costs (constant $8 per unit per month). The inventory balance equations ensure that inventory from the previous month plus production minus demand equals current inventory. This allows Acme to produce in low-cost months and store for high-demand months. Non-negativity ensures feasible solutions."""
        ))

        examples.append(TrainingExample(
            instruction="Formulate a transportation problem",
            input="""A company has 3 factories with supplies of 100, 150, and 200 units respectively. 
There are 4 warehouses with demands of 120, 80, 150, and 100 units respectively.
Transportation costs from factory i to warehouse j are given in a cost matrix.
Minimize total transportation cost.""",
            output="""**Problem Analysis:**
This is a transportation problem where we need to ship goods from factories (sources) to warehouses (destinations) at minimum cost.

**Decision Variables:**
- x_{ij} = amount shipped from factory i to warehouse j (nonnegative), i = 1, 2, 3; j = 1, 2, 3, 4

**Given Data:**
- Supplies: s_1 = 100, s_2 = 150, s_3 = 200 (units)
- Demands: d_1 = 120, d_2 = 80, d_3 = 150, d_4 = 100 (units)
- Transportation costs: c_{ij} = cost from factory i to warehouse j (dollars per unit), given in cost matrix

**Objective (minimize total transportation cost):**
min Z = Σ_{i=1}^3 Σ_{j=1}^4 c_{ij} x_{ij}

**Mathematical Formulation:**

$$
\\begin{aligned}
\\text{Minimize} \\quad & \\sum_{i=1}^{3} \\sum_{j=1}^{4} c_{ij} x_{ij} \\\\
\\text{Subject to} \\quad & \\sum_{j=1}^{4} x_{ij} \\le s_i, \\quad \\forall i = 1, 2, 3 \\\\
& \\sum_{i=1}^{3} x_{ij} \\ge d_j, \\quad \\forall j = 1, 2, 3, 4 \\\\
& x_{ij} \\ge 0, \\quad \\forall i = 1, 2, 3, \\forall j = 1, 2, 3, 4
\\end{aligned}
$$

**Explanation:**
The objective minimizes total transportation costs. Supply constraints ensure each factory doesn't ship more than available. Demand constraints ensure each warehouse receives at least its required amount."""
        ))
        

        examples.append(TrainingExample(
            instruction="Formulate an assignment problem",
            input="Assign 4 workers to 4 tasks. Each worker can do each task with different costs c_ij. Each worker must be assigned to exactly one task, and each task must be done by exactly one worker. Minimize total cost.",
            output="""**Variable Definitions:**
- x_{ij} = 1 if worker i is assigned to task j, 0 otherwise, i = 1, 2, 3, 4; j = 1, 2, 3, 4
- c_{ij} = cost of assigning worker i to task j ($), i = 1, 2, 3, 4; j = 1, 2, 3, 4 (given)

**Mathematical Formulation:**

$$
\\begin{aligned}
\\text{Minimize} \\quad & \\sum_{i=1}^{4} \\sum_{j=1}^{4} c_{ij} x_{ij} \\\\
\\text{Subject to} \\quad & \\sum_{j=1}^{4} x_{ij} = 1, \\quad \\forall i = 1, 2, 3, 4 \\\\
& \\sum_{i=1}^{4} x_{ij} = 1, \\quad \\forall j = 1, 2, 3, 4 \\\\
& x_{ij} \\in \\{0, 1\\}, \\quad \\forall i = 1, 2, 3, 4, \\forall j = 1, 2, 3, 4
\\end{aligned}
$$

**Explanation:**
The objective minimizes total assignment cost. The first set of constraints ensures each worker is assigned to exactly one task. The second set ensures each task is assigned to exactly one worker. Binary constraints ensure assignments are either made (1) or not (0).

**Feasibility Notes:**
- Problem is always feasible for square assignment problems (equal number of workers and tasks)
- The assignment constraints are always binding
- This is a special case of transportation problem with unit supplies and demands"""
        ))
        
        return examples
    
    def create_incorrect_example_corrections(self) -> List[TrainingExample]:
        """
        Create examples that show common mistakes and their corrections.
        This helps the model avoid common errors.
        """
        examples = []

        examples.append(TrainingExample(
            instruction="Formulate a production planning problem with inventory",
            input="""Acme Manufacturing needs to deliver windows over 6 months.
Demands: Month 1: 100, Month 2: 250, Month 3: 190, Month 4: 140, Month 5: 220, Month 6: 110
Production costs per window: Month 1: 50, Month 2: 45, Month 3: 55, Month 4: 48, Month 5: 52, Month 6: 50
Storage cost: $8 per window per month (end-of-month inventory)
Initial inventory I_0 = 0. Final inventory I_6 = 0 (no leftover at end of horizon).
Minimize total cost.""",
            output="""**Decision Variables (for months t = 1, ..., 6):**
- p_t = units produced in month t (nonnegative)
- I_t = end-of-month inventory after month t (nonnegative)

**Given Data:**
- Demands: d = [100, 250, 190, 140, 220, 110]
- Production costs: c^(p) = [50, 45, 55, 48, 52, 50] (dollars per window)
- Holding cost: h = 8 dollars per window per month
- Initial inventory I_0 = 0
- Final inventory forced to zero: I_6 = 0 (no leftover at end of horizon)

**Objective (minimize total production + holding cost):**
min Z = Σ_{t=1}^6 c_t^(p) p_t + 8 Σ_{t=1}^6 I_t

**Mathematical Formulation:**

$$
\\begin{aligned}
\\text{Minimize} \\quad & \\sum_{t=1}^{6} c_t^{(p)} p_t + 8 \\sum_{t=1}^{6} I_t \\\\
\\text{Subject to} \\quad & I_0 + p_1 - d_1 = I_1 \\quad \\text{(with } I_0 = 0\\text{)} \\\\
& I_{t-1} + p_t - d_t = I_t, \\quad \\forall t = 2, \\ldots, 6 \\\\
& I_6 = 0 \\quad \\text{(terminal inventory constraint)} \\\\
& p_t \\ge 0, \\quad I_t \\ge 0, \\quad \\forall t = 1, \\ldots, 6
\\end{aligned}
$$

**Explanation:**
This is a standard linear program (continuous variables). The objective function separates production costs (which vary by period) from holding costs (constant per unit per period). The inventory balance equations I_{t-1} + p_t - d_t = I_t model the flow of inventory from period to period. The terminal constraint I_6 = 0 ensures all demand is met with no leftover inventory. This formulation is mathematically correct and complete."""
        ))

        examples.append(TrainingExample(
            instruction="Formulate a production planning problem with inventory",
            input="""A company needs to meet demand for a product over 4 months.
Demands: Month 1: 200, Month 2: 300, Month 3: 250, Month 4: 350
Production costs: Month 1: 10, Month 2: 12, Month 3: 11, Month 4: 13 (dollars per unit)
Holding cost: 2 dollars per unit per month
Starting inventory is zero. All demand must be met exactly with no leftover inventory.
Minimize total cost.""",
            output="""**Decision Variables (for months t = 1, ..., 4):**
- p_t = units produced in month t (nonnegative)
- I_t = end-of-month inventory after month t (nonnegative)

**Given Data:**
- Demands: d = [200, 300, 250, 350]
- Production costs: c^(p) = [10, 12, 11, 13] (dollars per unit)
- Holding cost: h = 2 dollars per unit per month
- Initial inventory I_0 = 0
- Final inventory I_4 = 0 (all demand must be met exactly, no leftover)

**Objective (minimize total production + holding cost):**
min Z = Σ_{t=1}^4 c_t^(p) p_t + 2 Σ_{t=1}^4 I_t

**Mathematical Formulation:**

$$
\\begin{aligned}
\\text{Minimize} \\quad & \\sum_{t=1}^{4} c_t^{(p)} p_t + 2 \\sum_{t=1}^{4} I_t \\\\
\\text{Subject to} \\quad & I_0 + p_1 - d_1 = I_1 \\quad \\text{(with } I_0 = 0\\text{)} \\\\
& I_{t-1} + p_t - d_t = I_t, \\quad \\forall t = 2, \\ldots, 4 \\\\
& I_4 = 0 \\quad \\text{(terminal inventory constraint - all demand met, no leftover)} \\\\
& p_t \\ge 0, \\quad I_t \\ge 0, \\quad \\forall t = 1, \\ldots, 4
\\end{aligned}
$$

**Explanation:**
The problem states "all demand must be met exactly with no leftover inventory", which requires the terminal constraint I_4 = 0. The objective function correctly separates variable production costs from constant holding costs. The inventory balance equations ensure period-by-period inventory flow."""
        ))
        
        return examples
    
    def save_training_data(self, examples: List[TrainingExample], filename: str = "training_data.jsonl"):
        """
        Save training examples in JSONL format for fine-tuning.
        
        Args:
            examples: List of training examples
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example.to_mistral_format(), f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(examples)} training examples to {output_path}")
        return output_path
    
    def create_training_dataset(self) -> Path:
        """
        Create complete training dataset with correct examples and corrections.
        
        Returns:
            Path to the saved training data file
        """
        logger.info("Creating training dataset...")
        
        correct_examples = self.create_correct_formulation_examples()
        logger.info(f"Created {len(correct_examples)} correct formulation examples")
        
        correction_examples = self.create_incorrect_example_corrections()
        logger.info(f"Created {len(correction_examples)} correction examples")
        
        all_examples = correct_examples + correction_examples
        
        output_path = self.save_training_data(all_examples)
        
        logger.info(f"Total training examples: {len(all_examples)}")
        logger.info(f"Training data saved to: {output_path}")
        
        return output_path


def create_finetuning_script(output_dir: Path = Path("finetuning_data")):
    """
    Create a script for fine-tuning using Hugging Face Transformers.
    
    Args:
        output_dir: Directory containing training data
    """
    script_content = """#!/usr/bin/env python3
\"\"\"
Fine-tuning script for Mistral-7B-Instruct-v0.2 using Hugging Face Transformers.
Run this script to fine-tune the model on your training data.

Requirements:
- pip install transformers datasets accelerate peft bitsandbytes
- Sufficient GPU memory (16GB+ recommended for 7B model)
- Training data in JSONL format (created by FineTuningDataPreparer)
\"\"\"

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAINING_DATA_PATH = "finetuning_data/training_data.jsonl"
OUTPUT_DIR = "finetuned_model"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4

def load_training_data(data_path: str):
    \"\"\"Load training data from JSONL file.\"\"\"
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def format_prompt(example):
    \"\"\"Format example for training.\"\"\"
    messages = example['messages']
    
    # Format as Mistral chat template
    formatted = ""
    for msg in messages:
        if msg['role'] == 'system':
            formatted += f"<s>[INST] {msg['content']} [/INST]"
        elif msg['role'] == 'user':
            formatted += f" {msg['content']} [/INST]"
        elif msg['role'] == 'assistant':
            formatted += f" {msg['content']} </s>"
    
    return {"text": formatted}

def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for memory efficiency
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Setup LoRA for efficient fine-tuning
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load training data
    print(f"Loading training data from {TRAINING_DATA_PATH}...")
    training_examples = load_training_data(TRAINING_DATA_PATH)
    print(f"Loaded {len(training_examples)} examples")
    
    # Convert to dataset
    dataset = Dataset.from_list(training_examples)
    dataset = dataset.map(format_prompt)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=3,
        load_best_model_at_end=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
"""
    
    script_path = output_dir / "finetune.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"Fine-tuning script created at: {script_path}")
    return script_path


if __name__ == "__main__":
    preparer = FineTuningDataPreparer()
    training_data_path = preparer.create_training_dataset()
    
    script_path = create_finetuning_script()
    
    print("\n" + "="*60)
    print("Fine-tuning Setup Complete!")
    print("="*60)
    print(f"Training data: {training_data_path}")
    print(f"Fine-tuning script: {script_path}")
    print("\nTo fine-tune the model:")
    print(f"1. Install requirements: pip install transformers datasets accelerate peft bitsandbytes")
    print(f"2. Run: python {script_path}")
    print("="*60)

