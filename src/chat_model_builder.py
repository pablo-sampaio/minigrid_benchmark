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
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # better than "fp4" for virtually all LLM use cases
                    bnb_4bit_compute_dtype=compute_dtype,
                    # Double-quantization (when enabled): quantize the quantization constants themselves, saving ~0.4 bits/param 
                    # (with near-zero quality cost, but with a slight loss in compute time). Disabled.
                    bnb_4bit_use_double_quant=False,
                )
            else:
                raise ValueError(f"Unsupported quantization type for HF: {hf_quantization!r}. Use '8bit', '4bit' or None.")

            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        # ------------------------------------------------------------------
        # Multimodal and task normalization support
        # Some causal text-only models are tagged by registry heuristics as
        # "text2text-generation", which would route them to Seq2Seq loaders
        # and fail. Preserve true multimodal task routing, and normalize
        # non-encoder-decoder text models to "text-generation".
        # ------------------------------------------------------------------
        _TEXT_COMPATIBLE_TASKS = {"text-generation", "image-text-to-text"}

        from transformers import AutoConfig
        from transformers.pipelines import get_task

        detected_task = get_task(model_id, token=api_key)

        if detected_task == "text2text-generation":
            config = AutoConfig.from_pretrained(model_id, token=api_key, trust_remote_code=True)
            if not getattr(config, "is_encoder_decoder", False):
                detected_task = "text-generation"
        elif detected_task not in _TEXT_COMPATIBLE_TASKS:
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

