# BLAT

This repository provides codes and checkpoints for extracting audio and text representations using BLAT (**B**ootstrapping **L**anguage-**A**udio pre-training based on **T**ag-guided synthetic data) models.

## Inference

First install the missing dependencies: `pip install -r requirements`. Then download the pre-trained weights:
```bash
$ wget https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.2/contrastive_pretrain_cnn14_bertm.pth -O checkpoints/blat_cnn14_bertm/model.pth
```

Refer to `inference.py` for the usage:
```python```
from inference import load_blat, encode_audio, encode_text
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio = "./example.wav"
text = ['a dog barks', 'a man is speaking', 'birds are chirping']
with torch.no_grad():
    audio_emb = encode_audio(model, audio, device)
    text_emb = encode_text(model, text_tokenizer, text, device, max_length)
sim = np.matmul(audio_emb, text_emb.T)
print(sim) # [[[[0.56612206 0.18251741 0.15569025]]
```

## Citation

