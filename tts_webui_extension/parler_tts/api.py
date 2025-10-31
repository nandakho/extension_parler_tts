import torch
import os

from transformers import AutoTokenizer, AutoFeatureExtractor

from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.manage_model_state import manage_model_state

device = "cuda:0" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler-tts-mini-v1"
repo_id_large = "ylacombe/parler-large-v1-og"

feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
SAMPLE_RATE = feature_extractor.sampling_rate

LOCAL_DIR = os.path.join("data", "models", "parler_tts")
LOCAL_MODEL_DIR = os.path.join(LOCAL_DIR, "cache")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


@manage_model_state(model_namespace="parler_tts")
def get_parler_tts_model(
    model_name=repo_id, attn_implementation=None, compile_mode=None
):
    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=LOCAL_MODEL_DIR,
        attn_implementation=attn_implementation,
        # attn_implementation = "eager" # "sdpa" or "flash_attention_2"
        revision="refs/pr/9" if model_name == repo_id_large else None,
    ).to(device)

    if compile_mode is not None:
        # compile_mode = "default"  # chose "reduce-overhead" for 3 to 4x speed-up
        model.generation_config.cache_implementation = "static"
        # compile the forward pass
        model.forward = torch.compile(model.forward, mode=compile_mode)

    return model


@manage_model_state(model_namespace="parler_tts_tokenizer")
def get_tokenizer(model_name=repo_id):
    return AutoTokenizer.from_pretrained(model_name, cache_dir=LOCAL_MODEL_DIR)


def tts(text, description, model_name, attn_implementation=None, compile_mode=None):
    tokenizer = get_tokenizer(repo_id)
    inputs = tokenizer(description.strip(), return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    model = get_parler_tts_model(
        model_name, attn_implementation=attn_implementation, compile_mode=compile_mode
    )

    generation = model.generate(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        do_sample=True,
        temperature=1.0,
    )

    return {"audio_out": (SAMPLE_RATE, generation.squeeze(0).cpu().numpy())}
