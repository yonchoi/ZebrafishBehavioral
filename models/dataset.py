from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """ Custom dataset for inputting continuous data"""

    def __init__(self, input_ids, attention_mask):
        self.input_ids  = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = {"input_ids"  : self.input_ids[idx],
                  "attention_mask" : self.attention_mask[idx]}

        return sample
