"""
Quick script to generate training data for fine-tuning.
Run this to create the training dataset.
"""

from src.utils.finetuning import FineTuningDataPreparer, create_finetuning_script

if __name__ == "__main__":
    print("="*60)
    print("Generating Training Data for Fine-Tuning")
    print("="*60)
    
    preparer = FineTuningDataPreparer()
    training_data_path = preparer.create_training_dataset()
    
    script_path = create_finetuning_script()
    
    print("\n" + "="*60)
    print("Training Data Generation Complete!")
    print("="*60)
    print(f"✓ Training data: {training_data_path}")
    print(f"✓ Fine-tuning script: {script_path}")
    print("\nNext steps:")
    print("1. Review the training data in finetuning_data/training_data.jsonl")
    print("2. Add more examples if needed")
    print("3. Install requirements: pip install transformers datasets accelerate peft bitsandbytes")
    print(f"4. Run fine-tuning: python {script_path}")
    print("\nSee FINETUNING_GUIDE.md for detailed instructions.")
    print("="*60)

