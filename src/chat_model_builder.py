import os
from typing import Any
import warnings

from langchain.chat_models.base import BaseChatModel


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")


def resolve_api_key(provider: str, explicit_api_key: str | None) -> str | None:
    if explicit_api_key:
        return explicit_api_key

    if provider == "openai":
        key_options = ["OPENAI_API_KEY", "OPENAI_KEY"]

    elif provider == "deepseek":
        key_options = ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"]

    elif provider == "hf":
        key_options = ["HF_API_KEY", "HF_TOKEN"]

    else:
        warnings.warn(f"Provider '{provider}' is not recognized for API key resolution.")
        return None

    for key in key_options:
        api_key = os.getenv(key)
        if api_key:
            return api_key
   
    warnings.warn(f"API key not found for provider '{provider}'.")
    return None


def build_chat_model(provider: str, model_id: str, api_key: str|None, max_output_tokens: int, hf_quantization: str|None) -> BaseChatModel:

    if provider == "openai":
        if not api_key:
            raise ValueError("API key not found for OpenAI. Set OPENAI_API_KEY or use --api-key.")

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, api_key=api_key, max_completion_tokens=max_output_tokens, max_retries=5)

    if provider == "deepseek":
        if not api_key:
            raise ValueError("API key not found for DeepSeek. Set DEEPSEEK_API_KEY or use --api-key.")
        
        from langchain_deepseek import ChatDeepSeek
        
        # Obs.: DeepSeek seems to use 'max_tokens' parameter (deprecated in OpenAI' API) for the number of generated tokens
        return ChatDeepSeek(model=model_id, api_key=api_key, max_tokens=max_output_tokens, max_retries=5)

    if provider == "hf":
        if not api_key:
            raise ValueError("API key not found for HuggingFace. Set HF_API_KEY or HF_TOKEN or use --api-key.")

        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import BitsAndBytesConfig

        # Ensure HF auth is available to transformers in Colab/Kaggle/local runs.
        os.environ.setdefault("HF_TOKEN", api_key)

        model_kwargs = {"token": api_key}
        if hf_quantization not in (None, "none"):
            import torch
            hf_quantization = hf_quantization.lower()
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if hf_quantization in ("8bit", "8bits"):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif hf_quantization in ("4bit", "4bits", "nf4", "fp4"):
                quant_type = "nf4" if hf_quantization in ("4bit", "nf4") else "fp4"
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            else:
                raise ValueError(f"Unsupported quantization type for HF: {hf_quantization}. Use '8bit', '4bit', 'nf4', or 'fp4'.")
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "do_sample": True,
                # Avoid transformers warning when models ship with default max_length.
                "max_length": None,
                "max_new_tokens": max_output_tokens,
                "return_full_text": False,
            },
            model_kwargs=model_kwargs,
        )

        return ChatHuggingFace(llm=hf_pipeline, max_retries=5)

    raise ValueError(f"Invalid provider: {provider}. Accepted values: {SUPPORTED_PROVIDERS}")
