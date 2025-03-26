
from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch

def save_mixed_tokens(
        pretrain_dataset_name: str = "dclm-200M_SmolLM2-1.7B",
        sft_dataset_name: str = "OpenR1-Math-220k-200M_SmolLM2-1.7B",
        ):
    data_path = "/home/user/repos/ModelDiffing/data/" 
    # load pretraining dataset
    try:
         pretrain_dataset = load_from_disk(data_path + pretrain_dataset_name)
    except Exception as e:
                print(f"Couldnt load pretraining dataset from disk: {e}")

    # load sft dataset
    try:
         sft_dataset = load_from_disk(data_path + sft_dataset_name)
    except Exception as e:
                print(f"Couldnt load SFT dataset from disk: {e}")
    print("Loaded both datasets from disk")

    combined_ds = concatenate_datasets([pretrain_dataset, sft_dataset])
    shuffled_ds = combined_ds.shuffle(seed=42)
    all_tokens = shuffled_ds["tokenized"]
    attention_masks = shuffled_ds["attention_mask"]
    # this is a list of lists, we want a pytorch tensor instead. torch.cat(all_tokens) complains that we want it to be a list of tensors, so we instead do
    all_tokens = torch.tensor(all_tokens)
    attention_masks = torch.tensor(attention_masks)
    print("Now saving both tensors")
    torch.save(all_tokens, "/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M.pt")
    torch.save(attention_masks, "/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M_am.pt")
    return all_tokens, attention_masks

def load_mixed_tokens(
            # base_model_id,
            pretrain_dataset_name: str = "dclm-200M_SmolLM2-1.7B",
            sft_dataset_name: str = "OpenR1-Math-220k-200M_SmolLM2-1.7B",
            ):
    try:
        all_tokens = torch.load("/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M.pt")
        attention_masks = torch.load("/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M_am.pt")
    except:
        print("Data is not cached. Shuffling and saving data")
        all_tokens, attention_masks = save_mixed_tokens(
              pretrain_dataset_name,
              sft_dataset_name,
              )
    # all_tokens = all_tokens.to("cuda")
    # attention_masks = attention_masks.to("cuda")
    return all_tokens, attention_masks

if __name__ == "__main__":
     save_mixed_tokens()