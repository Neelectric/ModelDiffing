# Model Diffing
This repository includes code for Model Diffing experiments. 

# Reproducing my results
Inspired by [open-r1], I make use of uv for this project. Create a virtual environment with uv via <uv venv --python 3.10 && source venv/bin/activate && uv pip install --upgrade pip && uv pip install -r requirements.txt>
May also need <uv pip install -q git+https://github.com/Neelectric/TransformerLensQwen2.5.git@main>

# Repo structure
- In ./cc_train, I organise all code relating to training cross-coders. This borrows lots of code from the [open-source replication repo](https://github.com/ckkissane/crosscoder-model-diff-replication) by Kissane et al. and an [open-source repo](https://github.com/neelnanda-io/Crosscoders) by Neel Nanda.
- In ./data_exp, I store all Python notebooks that I use for data exploration.
- In ./data, I store all datasets after pre-processing. Following Kissane et al., these are often pytorch tensors of input_ids.
- In ./auto_interp, I organise all code relating to automatically interpreting cross-coder features. This leans heavily on approaches by [EleutherAI](https://github.com/EleutherAI/delphi).


## Credits
This repository utilizes [cross-coders](https://transformer-circuits.pub/2024/crosscoders/index.html) as introduced by Anthropic for [model diffing](https://transformer-circuits.pub/2024/model-diffing/index.html). It builds upon an [open-source replication repo](https://github.com/ckkissane/crosscoder-model-diff-replication) by Connor Kissane, which itself extends an [open-source repo](https://github.com/neelnanda-io/Crosscoders) by Neel Nanda. Further, it is informed by the resulting [less-wrong blogpost](https://www.lesswrong.com/posts/srt6JXsRMtmqAJavD/open-source-replication-of-anthropic-s-crosscoder-paper-for#Implementation_details_and_tips_for_training_and_analysis) by Connor Kissane, Robert Krzyzanowski, Arthur Conmy and Neel Nanda.