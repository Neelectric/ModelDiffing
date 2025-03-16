import torch 
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm, trange
from huggingface_hub import hf_api
from tempfile import NamedTemporaryFile


def tokenize_and_truncate(formatted_instances):
    tokenized = tokenizer(formatted_instances, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    detokenized = tokenizer.batch_decode(tokenized["input_ids"])
    return tokenized, detokenized

def sample_half_toks(tulu, max_num_tokens):
    raw_instances = []
    formatted_conversations = []
    tokenized_conversations = []
    num_tokens = 0
    ds_columns = tulu.column_names
    formatted_ds = tulu.map(lambda x: {"formatted_chat" : tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}, num_proc=16, remove_columns=ds_columns)
    progress_bar = tqdm(total=max_num_tokens, desc="Tokens processed", unit="token")
    for i in range(0, len(tulu), 100):
        instances = formatted_ds[i:min(i+100, len(tulu))]["formatted_chat"]
        tokenized, detokenized = tokenize_and_truncate(instances)
        len_tokenized = tokenized["input_ids"].shape[0] * tokenized["input_ids"].shape[1]
        if num_tokens + len_tokenized > max_num_tokens:
            break
        raw_instances.extend(instances)
        formatted_conversations.extend(detokenized)
        tokenized_conversations.extend(tokenized["input_ids"])
        num_tokens += len_tokenized
        progress_bar.update(len_tokenized)
    progress_bar.close()
    return raw_instances, formatted_conversations, tokenized_conversations, num_tokens

def create_dataset(raw_instances, formatted_conversations, tokenized_conversations):
    assert len(raw_instances) == len(formatted_conversations) == len(tokenized_conversations)
    tokenized_conversations_tensor = torch.stack(tokenized_conversations)
    dataset_list = [
        {
            "raw": raw,
            "formatted": formatted,
            "tokenized": tokenized,
        }
        for raw, formatted, tokenized in zip(
            raw_instances, formatted_conversations, tokenized_conversations_tensor
        )
    ]
    return Dataset.from_list(dataset_list)

if __name__ == "__main__":
    max_tokens = 400_000_000
    model_id = "allenai/OLMo-2-1124-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    max_length = 4096
    half_max_tokens = max_tokens // 2
    print(f"max_tokens is {max_tokens}, so we will sample {half_max_tokens}")

    dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture", "default")["train"]
    tulu = dataset.shuffle(seed=42)
    raw_instances, formatted_conversations, tokenized_conversations, num_tokens = sample_half_toks(tulu, half_max_tokens)

    dataset = create_dataset(raw_instances, formatted_conversations, tokenized_conversations)
 

    print(f"Generated data with {num_tokens} tokens")
    print(f"Saving generated data to disk")

    dataset.save_to_disk("/home/user/repos/ModelDiffing/data/tulu_200M-OLMo-2")
    dataset.push_to_hub("Neelectric/tulu_200M-OLMo-2")