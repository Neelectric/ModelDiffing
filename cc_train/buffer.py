# Written by Neel Rajani, 05.03.25. Directly adapted from https://github.com/ckkissane/crosscoder-model-diff-replication

from utils import *
from transformer_lens import ActivationCache
import tqdm

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens, attention_masks):
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        self.attention_masks = attention_masks
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        self.cfg["estimated_norm_scaling_factor_A"] = estimated_norm_scaling_factor_A
        self.cfg["estimated_norm_scaling_factor_B"] = estimated_norm_scaling_factor_B
        print(f"Estimated norm scaling factor A: {estimated_norm_scaling_factor_A}")
        print(f"Estimated norm scaling factor B: {estimated_norm_scaling_factor_B}")
        
        self.normalisation_factor = torch.tensor(
        [
            estimated_norm_scaling_factor_A,
            estimated_norm_scaling_factor_B,
        ],
        # device="cuda:0",
        device=cfg["device"],
        dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        model_device = model.cfg.device
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            attention_mask = self.attention_masks[i * batch_size : (i + 1) * batch_size]

            # Create properly formatted inputs dictionary
            inputs = {
                "input_ids": tokens.to(model_device),
                "attention_mask": attention_mask.to(model_device)
            }
            # torch.cuda.empty_cache() 
            _, cache = model.run_with_cache(
                inputs,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            # del tokens, attention_mask
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
            # del _, cache
            # torch.cuda.empty_cache() 
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast(device, torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"], dynamic_ncols=True):
                tokens = self.all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"], num_batches
                    )
                ]
                attention_mask = self.attention_masks[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"], num_batches
                    )
                ]
                inputs = {
                "input_ids": tokens,
                "attention_mask": attention_mask
                }

                _, cache_A = self.model_A.run_with_cache(
                    inputs, names_filter=self.cfg["hook_point"]
                )
                cache_A: ActivationCache

                _, cache_B = self.model_B.run_with_cache(
                    inputs, names_filter=self.cfg["hook_point"]
                )
                cache_B: ActivationCache
                # cache_B = cache_B.to(self.model_A.cfg.device)
                cache_B_acts = cache_B[self.cfg["hook_point"]].to(self.model_A.cfg.device)

                acts = torch.stack([cache_A[self.cfg["hook_point"]], cache_B_acts], dim=0)
                acts = acts[:, :, 1:, :] # Drop BOS
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # [2, batch, seq_len, d_model]
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out