
from utils import *
from trainer import Trainer
from transformers import AutoModelForCausalLM
from data_scripts.prepare_alltokens import load_mixed_tokens

# device = 'cuda:0'
device = "cpu"
n_devices = 1
if torch.cuda.is_available():
    device = "cuda"
    n_devices = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    device = "mps"

### Gemma ids
# base_model_id = "google/gemma-2-2b"
# ft_model_id = "google/gemma-2-2b-it"

### Qwen ids
base_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# ft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ft_model_id = "Neelectric/SmolLM2-1.7B-Instruct_SFT"

### load in the tokens
all_tokens, attention_masks = load_mixed_tokens(
            # base_model_id,
            pretrain_dataset_name = "dclm-200M_SmolLM2-1.7B",
            sft_dataset_name = "OpenR1-Math-220k-200M_SmolLM2-1.7B",
            )

### load in the models from hf so we can use FA2 for SWA, then pass to TransformerLens
base_model_hf = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    ).to("cuda")
ft_model_hf = AutoModelForCausalLM.from_pretrained(
    ft_model_id, 
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    ).to("cuda")
base_model = HookedTransformer.from_pretrained(
    base_model_id, 
    # device=device, 
    # n_devices=n_devices,
    # from_pretrained_kwargs={"attn_implementation":"flash_attention_2",},
    hf_model=base_model_hf,
    fold_value_biases=False,
)

ft_model = HookedTransformer.from_pretrained(
    ft_model_id, 
    # device=device, 
    # n_devices=n_devices,
    # from_pretrained_kwargs={"attn_implementation":"flash_attention_2",},
    hf_model=ft_model_hf,
    fold_value_biases=False,
)

default_cfg = {
    "seed": 42,
    "batch_size": 4096, #originally 4096
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000, #originally 400_000_000
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 8192,
    "enc_dtype": "fp32",
    "model_name": "SmolLM2-1.7B-Instruct",
    "site": "resid_pre",
    "device": device,
    "model_batch_size": 4,
    "log_every": 20,
    "save_every": 100, # originally 30000 
    "dec_init_norm": 0.08,
    "hook_point": "blocks.12.hook_resid_pre",
    "wandb_project": "R1-crosscoder",
    "wandb_entity": "Neelectric",
    "run_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct_vs_SFT_crosscoder",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, ft_model, all_tokens, attention_masks)
trainer.train()