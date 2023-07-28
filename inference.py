import argparse
from pathlib import Path

import librosa
from zsvision.zs_utils import load_json_config
import numpy as np
import torch
from transformers import AutoTokenizer

import utils


def load_model(config, ckpt_path, device):
    model = utils.init_obj_from_str(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def encode_audio(model, audio, device, sample_rate=32000):
    target_sr = model.audio_encoder.sample_rate
    if isinstance(audio, str) and Path(audio).exists():
        waveform = librosa.core.load(audio, sr=target_sr)[0]
    elif isinstance(audio, (np.ndarray, torch.Tensor)):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        waveform = librosa.core.resample(audio, sample_rate, target_sr)
    waveform = torch.as_tensor(waveform).to(device)
    audio_emb = model.encode_audio(**{
        "waveform": waveform.unsqueeze(0),
        "wave_length": torch.tensor([len(waveform)]),
    }).cpu().numpy()
    return audio_emb


def encode_text(model, text_tokenizer, text, device, max_length):
    if isinstance(text, str):
        text = [text]
    token = dict(text_tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True, return_tensors="pt")
    )
    for k, v in token.items():
        token[k] = v.to(device)
    text_emb = model.encode_text(**token)
    text_emb = text_emb.cpu().numpy()
    return text_emb


def load_blat(ckpt_dir, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(ckpt_dir)
    config = load_json_config(ckpt_dir / "config.json")

    model = load_model(config["model"], ckpt_dir / "model.pth", device)

    text_tokenizer = AutoTokenizer.from_pretrained(config["text_tokenizer"]["type"])
    max_length = config["text_tokenizer"]["max_length"]
    return model, text_tokenizer, max_length


def infer_sim(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, text_tokenizer, max_length = load_blat(args.ckpt_dir, device)
    
    with torch.no_grad():
        audio_emb = encode_audio(model, args.audio, device)
        text_emb = encode_text(model, text_tokenizer, args.text, device, max_length)

    at_sim = np.matmul(audio_emb, text_emb.T)

    print(f"audio: {args.audio}")
    for text, sim in zip(args.text, at_sim[0]):
        print(f"text: {text}, similarity: {sim:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--audio", type=str, default="")
    parser.add_argument("--text", type=str, nargs="+", default=[""])

    args = parser.parse_args()
    infer_sim(args)
