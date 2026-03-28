import os
import torch
import wandb
from loguru import logger
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
LORA_ADAPTER_DIR = "./models/exaone-lora-v2"
MERGED_OUTPUT_DIR = "./models/exaone-merged-v2"

def merge_lora():
    """Merges QLoRA adapter into the base EXAONE model."""
    logger.info("Starting LoRA merging process...")
    
    # Initialize wandb
    wandb.init(project="govon-qa", job_type="merge-lora")

    # Step 1: Load adapter config
    logger.info("[1/5] Loading adapter config...")
    adapter_config = PeftConfig.from_pretrained(LORA_ADAPTER_DIR)
    
    # Step 2: Load tokenizer
    logger.info("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # Step 3: Load base model in BF16
    logger.info("[3/5] Loading base model in BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Verify EXAONE embedding accessors
    try:
        base_model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        base_model.get_input_embeddings = lambda: base_model.transformer.wte
    try:
        base_model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        base_model.get_output_embeddings = lambda: base_model.lm_head

    # Step 4: Load adapter and merge
    logger.info("[4/5] Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model = model.merge_and_unload()

    # Step 5: Save merged model
    logger.info("[5/5] Saving merged model...")
    model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    logger.info(f"Successfully merged and saved to {MERGED_OUTPUT_DIR}")
    wandb.finish()

if __name__ == "__main__":
    merge_lora()
