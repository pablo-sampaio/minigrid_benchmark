import os

from langchain.chat_models.base import BaseChatModel


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")


def build_chat_model(provider: str, model_id: str, api_key: str|None, max_output_tokens: int, hf_quantization: str|None) -> BaseChatModel:

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, api_key=api_key, max_completion_tokens=max_output_tokens, max_retries=5)

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        
        # Obs.: DeepSeek seems to use 'max_tokens' parameter (deprecated in OpenAI' API) for the number of generated tokens
        return ChatDeepSeek(model=model_id, api_key=api_key, max_tokens=max_output_tokens, max_retries=5)

    if provider == "hf":
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
