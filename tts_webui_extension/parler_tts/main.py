import gradio as gr

from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.utils.randomize_seed import randomize_seed_ui
from tts_webui.utils.list_dir_models import model_select_ui, unload_model_button
from tts_webui.decorators.decorator_apply_torch_seed import decorator_apply_torch_seed
from tts_webui.decorators.decorator_log_generation import decorator_log_generation
from tts_webui.decorators.decorator_save_metadata import decorator_save_metadata
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.log_function_time import log_function_time

from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_outer,
    decorator_extension_inner,
)

from .api import tts


def extension__tts_generation_webui():
    main_ui()
    return {
        "package_name": "extension_parler_tts",
        "name": "Parler-TTS",
        "requirements": "git+https://github.com/rsxdalv/extension_parler_tts@main",
        "description": "Parler-TTS is a training and inference library for high-fidelity text-to-speech (TTS) models.",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/extension_parler_tts",
        "extension_website": "https://github.com/rsxdalv/extension_parler_tts",
        "extension_platform_version": "0.0.1",
    }


def verify_transformers_version():
    try:
        import transformers

        version = transformers.__version__
        if version != "4.46.1":
            return False
        return True
    except Exception:
        return False


repo_id = "parler-tts/parler-tts-mini-v1"
repo_id_large = "ylacombe/parler-large-v1-og"


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("parler_tts")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def generate_parler_tts(
    text,
    description,
    model_name,
    attn_implementation=None,
    compile_mode=None,
    **kwargs,
):
    return tts(
        text=text,
        description=description,
        model_name=model_name,
        attn_implementation=attn_implementation,
        compile_mode=compile_mode,
    )


def parler_tts_params_ui():
    text = gr.Textbox(
        label="Text",
        value="Hey, how are you doing today?",
    )
    description = gr.Textbox(
        label="Context",
        value="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    )

    return {text: "text", description: "description"}


version = "0.2.2"


def main_ui():
    gr.Markdown(
        f"""
        # Parler-TTS {version}
        Parler-TTS is a training and inference library for high-fidelity text-to-speech (TTS) models.

        More models can be found at: https://huggingface.co/models?filter=parler_tts

        Parler-TTS Large v1 has many issues, and is not recommended for use.

        Parler requires transformers==4.46.1: {"OK" if verify_transformers_version() else "NOT OK"}

        Download models using the Tools>Model Downloader. Then click refresh next to the model selector.
        """
    )

    with gr.Row():
        inner_ui()


def inner_ui():
    with gr.Column():
        parler_tts_params = parler_tts_params_ui()
        generate_button = gr.Button("Generate Audio", variant="primary")

        # with gr.Column():
        model_name = model_select_ui(
            [
                ("Parler-TTS Mini v1", repo_id),
                ("Parler-TTS Large v1", repo_id_large),
            ],
            "parler_tts",
        )

        # reload import randomize_seed_ui
        import importlib

        import tts_webui.utils.randomize_seed

        importlib.reload(tts_webui.utils.randomize_seed)
        from tts_webui.utils.randomize_seed import randomize_seed_ui

        seed, randomize_seed_callback = randomize_seed_ui()

        with gr.Row(
            variant="default",
            elem_classes=[
                # "bg-black-300",
                # align end
                # "justify-end",
                # "items-center justify-end",
                # "!items-end",
                "items-end",
            ],
        ):
            seed_input = gr.Textbox(label="Seed", value="-1")
            randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Row():
            attn_implementation = gr.Dropdown(
                choices=["eager", "sdpa", "flash_attention_2"],
                label="Attention Implementation",
                value="eager",
            )

            compile_mode = gr.Dropdown(
                choices=[("None", None), "default", "reduce-overhead"],
                label="Compile Mode",
                elem_classes=[
                    "bg-blue-100",
                    "text-black",
                    "font-mono",
                    "text-xs",
                    "p-0.5",
                    "rounded-md",
                    "height-10",
                ],
            )

        unload_model_button("parler_tts")

    with gr.Column():
        audio_out = gr.Audio(label="Parler-TTS generation", type="numpy")

    generate_button.click(**randomize_seed_callback).then(
        **dictionarize(
            fn=generate_parler_tts,
            inputs={
                **parler_tts_params,
                seed: "seed",
                model_name: "model_name",
                attn_implementation: "attn_implementation",
                compile_mode: "compile_mode",
            },
            outputs={"audio_out": audio_out},
        ),
        api_name="parler_tts",
    )


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()

    with gr.Blocks() as demo:
        main_ui()

    demo.launch()
