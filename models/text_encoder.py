from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Bert(nn.Module):

    def __init__(self, model_type):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_type)
        self.embed_dim = self.model.config.hidden_size

    def forward(self, **token):
        output = self.model(**token)
        # [CLS] pooling
        clip_emb = output.last_hidden_state[:, 0, :]
        time_emb = output.last_hidden_state
        output_dict = {
            "clip_emb": clip_emb,
            "time_emb": time_emb,
            # "time_mask": tokens["attention_mask"]
        }
        return output_dict


