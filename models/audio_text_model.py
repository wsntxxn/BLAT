from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"


class AudioTextClip(BaseModel):

    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 audio_dim,
                 text_dim,
                 shared_dim,
                 audio_forward_keys=["waveform", "wave_length"],
                 text_forward_keys=["input_ids", "token_type_ids",
                                    "attention_mask"],
                 ):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.audio_forward_keys = audio_forward_keys
        self.text_forward_keys = text_forward_keys
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, input_dict):
        audio_input = {k: input_dict[k] for k in self.audio_forward_keys}
        audio_emb = self.encode_audio(**audio_input)
 
        text_input = {k: input_dict[k] for k in self.text_forward_keys}
        text_emb = self.encode_text(**text_input)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output

    
    def forward_multi_text(self, input_dict):
        batch_size = input_dict["waveform"].size(0)
        num_captions = input_dict["num_captions"]

        audio_input = {k: input_dict[k] for k in self.audio_forward_keys}
        audio_emb = self.encode_audio(**audio_input)
 
        text_input = {}
        for k in self.text_forward_keys:
            v = input_dict[k]
            text_input[k] = rearrange(text_input[k], "b n ... -> (b n) ...")
        text_emb = self.encode_text(**text_input)
        text_emb = rearrange(text_emb, "(b n) ... -> b n ...",
                             b=batch_size, n=num_captions)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output


    def evaluate_retrieval(self, input_dict):
        if "num_captions" in input_dict:
            return self.forward_multi_text(input_dict)
        else:
            return self.forward(input_dict)


    def encode_audio(self, **audio):
        audio_emb = self.audio_encoder(**audio)["clip_emb"]
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return audio_emb

    def encode_text(self, **text):
        text_emb = self.text_encoder(**text)["clip_emb"]
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return text_emb
