import os

from configs.first_full_config import first_full_config
from fine_tuning import run_fine_tuning

os.environ['HF_HOME'] = '/root/.cache/huggingface'

# Disable hf_transfer if not installed
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

run_fine_tuning(first_full_config)