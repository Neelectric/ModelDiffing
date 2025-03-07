# Written by Neel Rajani, 05.03.25. 

from datasets import load_dataset
import torch




def download_olmo_mix():
    dataset = load_dataset('allenai/olmo-mix-1124', split='train', streaming=True)
    # print(next(iter(dataset)))
    shuffled_dataset = dataset.shuffle(seed=41, buffer_size=10_000)
    print(len(dataset))
    print(shuffled_dataset)
    print(len(shuffled_dataset))
   
    # for _ in range(2):
    #     print(next(iter(dataset)))
    #     print(next(iter(dataset)))
    return


download_olmo_mix()