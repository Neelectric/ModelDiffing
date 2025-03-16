
from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch


# def shuffle_fineweb_openr1_mixed_tokens(base_model_id):
#     try:
#         fw_ds = load_from_disk("/home/user/repos/ModelDiffing/data/fineweb-qwen-2.5-math-1.5b")
#     except:
#         print("fineweb not found. Downloading")
#         fw_ds = load_dataset("Neelectric/fineweb-qwen-2.5-math-1.5b")
#     try:
#         or_ds = load_from_disk("/home/user/repos/ModelDiffing/data/openr1-math-220k-qwen-2.5-math-1.5b")
#     except:
#         print("or not found. Downloading")
#         or_ds = load_dataset("Neelectric/openr1-math-220k-qwen-2.5-math-1.5b")

#     combined_ds = concatenate_datasets([fw_ds, or_ds])
#     shuffled_ds = combined_ds.shuffle(seed=49)
#     all_tokens = shuffled_ds["tokenized"]
#     # this is a list of lists, we want a pytorch tensor instead. torch.cat(all_tokens) complains that we want it to be a list of tensors, so we instead do
#     all_tokens = torch.tensor(all_tokens)
#     torch.save(all_tokens, "/home/user/repos/ModelDiffing/data/fineweb-openr1-mixed-tokens.pt")
#     return all_tokens

# def load_fineweb_openr1_mixed_tokens(base_model_id):
#     try:
#         all_tokens = torch.load("/home/user/repos/ModelDiffing/data/fineweb-openr1-mixed-tokens.pt")
#     except:
#         print("Data is not cached. Shuffling and saving data")
#         all_tokens = shuffle_fineweb_openr1_mixed_tokens(base_model_id)
#     all_tokens = all_tokens.to("cuda")
#     return all_tokens


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

    combined_ds = concatenate_datasets([pretrain_dataset, sft_dataset])
    shuffled_ds = combined_ds.shuffle(seed=42)
    all_tokens = shuffled_ds["tokenized"]
    # this is a list of lists, we want a pytorch tensor instead. torch.cat(all_tokens) complains that we want it to be a list of tensors, so we instead do
    all_tokens = torch.tensor(all_tokens)
    torch.save(all_tokens, "/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M.pt")
    return

def load_mixed_tokens(
            # base_model_id,
            pretrain_dataset_name: str = "dclm-200M_SmolLM2-1.7B",
            sft_dataset_name: str = "OpenR1-Math-220k-200M_SmolLM2-1.7B",
            ):
    try:
        all_tokens = torch.load("/home/user/repos/ModelDiffing/data/mixed/dclm-openr1-mixed-tokens-400M.pt")
    except:
        print("Data is not cached. Shuffling and saving data")
        all_tokens = save_mixed_tokens(
              pretrain_dataset_name,
              sft_dataset_name,
              )
    all_tokens = all_tokens.to("cuda")
    return all_tokens

if __name__ == "__main__":
     save_mixed_tokens()