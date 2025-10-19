from config import HF_TOKEN
from prepare_voynich_dataset import generate_datasets
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset, val_dataset = generate_datasets('GC2a-n.txt', tokenizer, True)