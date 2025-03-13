# Written by Neel Rajani, 05.03.25. 

from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, get_dataset_config_names
import torch
from typing import Dict, List, Optional, Any
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
tokenizer.padding_side = "left"

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
    
    print(filtered_batch[0]["source"])
    # print(filtered_batch[0]["text"])
    
    return {
        'text': [item['text'] for item in filtered_batch],
        'source': [item['source'] for item in filtered_batch]
    }

def process_dataset(dataset_name: str = "allenai/dolmino-mix-1124", batch_size: int = 1, max_batches: int = 3):
    # create interleaved dataset with correct proportions
    proportion_dict = prepare_subset_mix(dataset_name)
    proportions = []
    dolmino_subsets = []
    for subset_name in proportion_dict.keys():
        subset = load_dataset(
            'allenai/dolmino-mix-1124',
            subset_name,
            split='train',
            streaming=True,
        )
        dolmino_subsets.append(subset)
        proportions.append(proportion_dict[subset_name])
    dataset = interleave_datasets(dolmino_subsets, probabilities=proportions, seed=42)
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    
    tokenized_batches = []
    lengths = torch.tensor([0], dtype=torch.float32)
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
            
        try:
            if not batch['text']:  # Skip empty batches
                print(f"Batch {i} is empty, skipping.")
                continue
            
            inputs = tokenizer(
                batch['text'],
                padding="max_length",
                truncation=True,
                max_length=4096,  
                return_tensors="pt"
            )
            
            input_ids = inputs.input_ids
            length = torch.tensor([input_ids.shape[1]], dtype=torch.float32)
            lengths = torch.cat((lengths, length))
            first_10_tokens = input_ids[0, :10]  
            decoded_tokens = tokenizer.convert_ids_to_tokens(first_10_tokens)
            
            tokenized_batches.append(inputs)
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
    print(lengths)
    mean = torch.mean(lengths)
    sum = torch.sum(lengths)
    print(f"Average token length: {mean}")
    print(f"Full token length: {sum}")
    
    return tokenized_batches

def process_math(batch_size=1, max_batches=10):
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split='train')
    counter = 0
    for elt in dataset:
        print(elt["messages"])
        print("\n\n")
        if counter == 5:
            break
        counter += 1
    return

def process_tulu(batch_size=1, max_batches=10):
    inst_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
    inst_tokenizer.padding_side = "left"
    dataset = load_dataset(
        "allenai/tulu-3-sft-olmo-2-mixture",
        split='train',
    )
    shuffled_dataset = dataset.shuffle(seed=42)
    counter = 0
    lengths = torch.tensor([0], dtype=torch.float32)
    for elt in tqdm(shuffled_dataset, total=max_batches, dynamic_ncols=True):
        # print(elt)

        messages = elt["messages"]
        templated = inst_tokenizer.apply_chat_template(messages, tokenize=False)
        # print(templated)
        tokenized = tokenizer(
            templated, 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=4096, 
            )
        # print(tokenized)
        # print("\n\n")
        input_ids = tokenized.input_ids
        print(input_ids) #in a normal convo this seems to have two instances of 100257
        length = torch.tensor([input_ids.shape[1]], dtype=torch.float32)
        lengths = torch.cat((lengths, length))




        if counter >= max_batches:
            break
        counter += 1
    print(lengths)
    mean = torch.mean(lengths)
    sum = torch.sum(lengths)
    print(f"Average token length: {mean}")
    print(f"Full token length: {sum}")
    
    return None

if __name__ == "__main__":
    print("Starting OLMo tokenization...")

    # process_dataset(batch_size=1, max_batches=2000)

    process_tulu(batch_size=1, max_batches=3)

    # process_math(batch_size=1, max_batches=10)

    # process_math()