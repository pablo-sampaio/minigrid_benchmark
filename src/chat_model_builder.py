import os

from langchain.chat_models.base import BaseChatModel


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")


def build_chat_model(
        provider: str,
        model_id: str,
        api_key: str | None,
        max_output_tokens: int,
        hf_quantization: str | None,
    ) -> BaseChatModel:

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            max_completion_tokens=max_output_tokens,
            max_retries=5,
        )

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        # DeepSeek uses the deprecated 'max_tokens' param for output length.
        return ChatDeepSeek(
            model=model_id,
            api_key=api_key,
            max_tokens=max_output_tokens,
            max_retries=5,
        )

    if provider == "hf":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import BitsAndBytesConfig

        # Ensure HF auth is available to transformers in Colab/Kaggle/local runs.
        os.environ.setdefault("HF_TOKEN", api_key)

        model_kwargs = {"token": api_key}

        if hf_quantization not in (None, "none"):
            import torch

            hf_quantization = hf_quantization.lower()

            # T4 does not support bfloat16 — force float16 unconditionally.
            # Even on capable GPUs, float16 is the safer default for bitsandbytes.
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            if hf_quantization in ("8bit", "8bits"):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif hf_quantization in ("4bit", "4bits"):
                quant_type = "nf4"  # better than "fp4" for virtually all LLM use cases in general
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    # Double-quantization: quantize the quantization constants
                    # themselves, saving ~0.4 bits/param (valuable on a T4), with near-zero quality cost.
                    bnb_4bit_use_double_quant=False, #True,
                )
            else:
                raise ValueError(f"Unsupported quantization type for HF: {hf_quantization!r}. Use '8bit', '4bit' or None.")

            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        # ------------------------------------------------------------------
        # Multimodal support
        # Models like LLaVA, Idefics, or Qwen-VL are registered as
        # "image-text-to-text" in the Transformers task registry and fail when
        # forced into a "text-generation" pipeline.  We detect the actual task
        # first and fall back to "text-generation" only when the model is
        # text-only.  The processor/tokenizer is loaded explicitly so that
        # multimodal tokenizers are handled correctly even though we feed the
        # pipeline plain text.
        # ------------------------------------------------------------------
        _TEXT_COMPATIBLE_TASKS = {"text-generation", "image-text-to-text", "text2text-generation"}

        from transformers.pipelines import get_task
        detected_task = get_task(model_id, token=api_key)

        if detected_task not in _TEXT_COMPATIBLE_TASKS:
            detected_task = "text-generation"

        # For multimodal models, the pipeline needs an AutoProcessor; for
        # text-only models an AutoTokenizer is sufficient.  Passing
        # `trust_remote_code` here covers custom-architecture models.
        if detected_task == "image-text-to-text":
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_id, token=api_key, trust_remote_code=True
            )
            model_kwargs["trust_remote_code"] = True
            extra_pipeline_kwargs: dict = {"tokenizer": processor.tokenizer}
        else:
            extra_pipeline_kwargs = {}

        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task=detected_task,
            pipeline_kwargs={
                "do_sample": True,
                # Avoid transformers warning when models ship with a default max_length.
                "max_length": None,
                "max_new_tokens": max_output_tokens,
                "return_full_text": False,
                **extra_pipeline_kwargs,
            },
            model_kwargs=model_kwargs,
        )

        return ChatHuggingFace(llm=hf_pipeline, max_retries=5)

    raise ValueError(f"Invalid provider: {provider!r}. Accepted values: {SUPPORTED_PROVIDERS}")

