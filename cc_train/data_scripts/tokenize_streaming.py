# Written by Neel Rajani, 05.03.25. 

from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, get_dataset_config_names, Dataset
import torch
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import os


def prepare_subset_mix(dataset_name: str = 'allenai/dolmino-mix-1124'):
    subsets = get_dataset_config_names(dataset_name)
    token_sizes = {
        "dclm": 752,
        "flan": 17,
        "pes2o": 58.6,
        "stackexchange": 1.26,
        "wiki": 3.7,
        "math": 10.7,
    }
    assert [list(token_sizes.keys())] == [subsets[1:]]
    total_size = sum(list(token_sizes.values()))
    proportion_dict = {
        key: round(token_sizes[key]/total_size, 5)
        for key in token_sizes.keys()
    }
    print(proportion_dict)
    return proportion_dict

def custom_collate_fn(batch):
    """
    Custom collation function that handles None values and only collates 'text' field.
    """
    # Filter out examples with None text
    filtered_batch = [item for item in batch if item is not None and item.get('text') is not None]
    if not filtered_batch:
        return {'text': []}
    if "source" in list(filtered_batch[0].keys()):
        return {
            'text': [item['text'] for item in filtered_batch],
            'source': [item['source'] for item in filtered_batch]
        }
    else:
        return {
            'text': [item['text'] for item in filtered_batch],
        }
    
def create_dataset(raw_instances, formatted_conversations, tokenized_conversations, attention_masks):
    assert len(raw_instances) == len(formatted_conversations) == len(tokenized_conversations) == len(attention_masks)
    tokenized_conversations_tensor = torch.stack(tokenized_conversations)
    dataset_list = [
        {
            "raw": raw,
            "formatted": formatted,
            "tokenized": tokenized,
            "attention_mask": attention_mask,
        }
        for raw, formatted, tokenized, attention_mask in zip(
            raw_instances, formatted_conversations, tokenized_conversations_tensor, attention_masks
        )
    ]
    return Dataset.from_list(dataset_list)

def process_pretrain_dataset(
        dataset_name: str = "allenai/dolmino-mix-1124", 
        subset_name: Optional[str] = None,
        batch_size: int = 1, 
        max_length: int = 4096,
        max_tokens: int = 200_000_000,
        ):
    if subset_name:
        dataset = load_dataset(
                dataset_name,
                subset_name,
                split='train',
                streaming=True,
            )
    # this creates an interleaved dataset with correct proportions (may be a more elegant way of doing this, but I'm not familiar)
    elif dataset_name == 'allenai/dolmino-mix-1124':
        proportion_dict = prepare_subset_mix(dataset_name)
        proportions = []
        dolmino_subsets = []
        for subset_name in proportion_dict.keys():
            subset = load_dataset(
                dataset_name,
                subset_name,
                split='train',
                streaming=True,
            )
            dolmino_subsets.append(subset)
            proportions.append(proportion_dict[subset_name])
        dataset = interleave_datasets(dolmino_subsets, probabilities=proportions, seed=42)
    else:
        dataset = load_dataset(
                dataset_name,
                split='train',
                streaming=True,
            )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    batches = []
    tokenized_batches = []
    attention_masks = []
    lengths = []
    tokens_so_far = 0
    with tqdm(total=max_tokens, dynamic_ncols=True) as pbar:
        for i, batch in enumerate(dataloader):
            try:
                if not batch['text']:  # Skip empty batches
                    print(f"Batch {i} is empty, skipping.")
                    continue
                
                inputs = tokenizer(
                    batch['text'],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,  
                    return_tensors="pt"
                )
                
                input_ids = inputs.input_ids
                length = count_non_pad_tokens(input_ids, tokenizer)
                lengths.append(lengths)
                tokens_so_far += length
                pbar.update(length)
                if tokens_so_far >= max_tokens:
                    print(f"Finished tokenizing {max_tokens} tokens. Exiting loop.")
                    break
                batches.append(batch['text'][0])
                tokenized_batches.append(inputs.input_ids[0])
                attention_masks.append(inputs.attention_mask[0])
                # print(tokenized_batches)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")

    mean = sum(lengths) / len(lengths)
    print(f"Average token length: {mean}")
    print(f"Full token length: {sum(lengths)}")
    # token_dataset = Dataset.from_list(tokenized_batches)
    token_dataset = create_dataset(batches, batches, tokenized_batches, attention_masks)
    pbar.close()
    return token_dataset


def tokenize_and_truncate(formatted_instances):
    tokenized = tokenizer(
        formatted_instances, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
        )
    detokenized = tokenizer.batch_decode(tokenized["input_ids"])
    return tokenized, detokenized


def count_non_pad_tokens(input_ids: torch.Tensor, tokenizer) -> int:
    # written by copilot
    """
    Counts the total number of tokens that are not the pad token in a [bsz, input_ids] tensor.

    Args:
        input_ids (torch.Tensor): A tensor of shape [bsz, seq_len] containing token IDs.
        tokenizer: The tokenizer used, which provides the pad token ID.

    Returns:
        int: The total number of non-pad tokens.
    """
    pad_token_id = tokenizer.pad_token_id
    non_pad_tokens = (input_ids != pad_token_id).sum().item()
    return non_pad_tokens

def sample_half_toks(
    SFT_dataset: Dataset, 
    max_num_tokens: int = 200_000_000, 
    batch_size: int = 100,  
                     ):
    raw_instances = []
    formatted_conversations = []
    tokenized_conversations = []
    attention_masks = []
    num_tokens = 0
    ds_columns = SFT_dataset.column_names
    formatted_ds = SFT_dataset.map(lambda x: {"formatted_chat" : tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}, num_proc=16, remove_columns=ds_columns)
    progress_bar = tqdm(total=max_num_tokens, desc="Tokens processed", unit="token")
    for i in range(0, len(SFT_dataset), 100):
        instances = formatted_ds[i:min(i+100, len(SFT_dataset))]["formatted_chat"]
        tokenized, detokenized = tokenize_and_truncate(instances)
        len_tokenized = count_non_pad_tokens(tokenized["input_ids"], tokenizer)
        num_tokens += len_tokenized
        if num_tokens > max_num_tokens:
            break
        raw_instances.extend(instances)
        formatted_conversations.extend(detokenized)
        tokenized_conversations.extend(tokenized["input_ids"])
        attention_masks.extend(tokenized["attention_mask"])
        progress_bar.update(len_tokenized)
    progress_bar.close()
    return raw_instances, formatted_conversations, tokenized_conversations, attention_masks, num_tokens

def process_SFT_dataset(
        dataset_name: str = "open-r1/OpenR1-Math-220k", 
        batch_size: int = 100, 
        max_length: int = 4096,
        max_tokens: int = 200_000_000,
        ):
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]
    shuffled_dataset = dataset.shuffle(seed=42)
    raw_instances, formatted_conversations, tokenized_conversations, attention_masks, num_tokens = sample_half_toks(shuffled_dataset, max_tokens, batch_size)
    dataset = create_dataset(raw_instances, formatted_conversations, tokenized_conversations, attention_masks)
    return dataset


if __name__ == "__main__":
    print("Starting tokenization...")
    dataset_type = "pretrain"
    # model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model_name = "Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer><|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # max_length = tokenizer.model_max_length
    max_length = 8192
    max_tokens = 200_000_000

    # dataset_name = "mlfoundations/dclm-baseline-1.0-parquet"
    # final_dataset_path = "dclm-200M_SmolLM2-1.7B"

    # dataset_name = "allenai/tulu-3-sft-olmo-2-mixture"
    # final_dataset_path = 
    
    dataset_name = "allenai/dolmino-mix-1124"
    subset_name = "pes2o"
    final_dataset_path = "dolmino-mix-1124_" + subset_name + "_" + model_name + "_200M-tokens"

    # dataset_name = "open-r1/OpenR1-Math-220k"
    # final_dataset_path = "OpenR1-Math-220k_" + model_name + "_200M-tokens"
    
    batch_size = 100
    save_directory = "/home/user/repos/ModelDiffing/data/" + final_dataset_path

    if dataset_type == "pretrain":
        processed_dataset = process_pretrain_dataset(
            dataset_name=dataset_name,
            subset_name=subset_name,
            batch_size=1, 
            max_length=max_length,
            max_tokens=max_tokens,
            )
    elif dataset_type == "SFT":
        assert subset_name == None
        processed_dataset = process_SFT_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size, 
            max_length=max_length,
            max_tokens=max_tokens,
            )

    # Save to disk and push to hub
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    processed_dataset.save_to_disk(save_directory)
    processed_dataset.push_to_hub("Neelectric/" + final_dataset_path)