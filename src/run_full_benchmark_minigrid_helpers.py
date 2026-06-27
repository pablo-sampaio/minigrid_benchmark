import os
import shutil
import sys
import warnings


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")

MODEL_OPTIONS = {
    "openai": [
        ("gpt-5.5", None), 
        ("gpt-5.4", None),
        ("gpt-5.4-mini", None),
        ("gpt-5.4-nano", None), 
        ("gpt-4.1", None),
        ("gpt-4.1-mini", None),
    ],
    "deepseek": [
        ("deepseek-v4-flash", None),
        ("deepseek-v4-pro", None),
    ],
    "hf": [
        ("google/gemma-3-4b-it", None),                 # quantizar?
        ("google/gemma-3-12b-it-qat-q4_0-gguf", None),
        ("google/gemma-4-E2B-it", None),
        ("google/gemma-4-12B-it-qat-q4_0-gguf", None),        
        ("Qwen/Qwen2.5-3B-Instruct", None),
        ("Qwen/Qwen2.5-7B-Instruct", "8bit"),
        ("Qwen/Qwen3-4B-Instruct-2507-FP8", None),
        ("Qwen/Qwen3-4B-Thinking-2507-FP8", None),
        ("WeiboAI/VibeThinker-3B", "8bit"),
    ],
}


def detect_execution_env() -> str:
    try:
        import kaggle_secrets  # noqa: F401

        return "kaggle"
    except ImportError:
        try:
            import google.colab  # noqa: F401

            return "colab"
        except ImportError:
            return "local"


def resolve_repo_path(execution_env: str, cwd: str | None = None) -> str:
    cwd = cwd or os.getcwd()
    if execution_env == "colab":
        return os.path.join(cwd, "minigrid_benchmark")
    if execution_env == "kaggle":
        return "/kaggle/working/minigrid_benchmark"

    candidates = [cwd, os.path.abspath(os.path.join(cwd, ".."))]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "src", "benchmark_minigrid.py")):
            return candidate

    return cwd


def clone_repo_if_needed(execution_env: str, repo_path: str, repo_url: str = "https://github.com/pablo-sampaio/minigrid_benchmark.git") -> bool:
    if execution_env not in ("colab", "kaggle") or os.path.exists(repo_path):
        return False

    os.system(f"git clone {repo_url} {repo_path}")
    return True


def append_src_to_syspath(repo_path: str) -> str:
    src_path = os.path.join(repo_path, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    return src_path


def configure_results_dir(execution_env: str, repo_path: str) -> str:
    if execution_env == "colab":
        from google.colab import drive
        drive.mount("/content/drive")
        results_dir = "/content/drive/My Drive/EAD-Pesquisa-Agentes/results"
    elif execution_env == "kaggle":
        results_dir = "/kaggle/working/results"
    else:
        results_dir = os.path.abspath(os.path.join(repo_path, "results"))

    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def resolve_api_key(provider: str, execution_env: str = "local") -> str | None:
    provider = provider.strip().lower()

    secret_names = None
    if provider == "openai":
        secret_names = ["OPENAI_API_KEY", "OPENAI_KEY"]
    if provider == "deepseek":
        secret_names = ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"]
    if provider == "hf":
        secret_names = ["HF_API_KEY", "HF_TOKEN"]

    if not secret_names:
        warnings.warn(f"Provider '{provider}' is not recognized for API key resolution.")
        return None

    if execution_env == "colab":
        from google.colab import userdata

        for secret_name in secret_names:
            try:
                api_key = userdata.get(secret_name)            
                if api_key:
                    return api_key
            except Exception:
                api_key = None

    elif execution_env == "kaggle":
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        for secret_name in secret_names:
            try:
                api_key = user_secrets.get_secret(secret_name)
                if api_key:
                    return api_key
            except Exception:
                api_key = None

    for secret_name in secret_names:
        api_key = os.getenv(secret_name)
        if api_key:
            return api_key

    warnings.warn(f"API key not found for provider '{provider}'.")
    return None


def create_model_selector_widgets(model_options: dict[str, list[tuple[str, str | None]]] | None = None):
    import ipywidgets as widgets

    model_options = model_options or MODEL_OPTIONS
    provider = list(model_options.keys())[0]
    model_id, quantization = model_options[provider][0]
    selection = {
        "provider": provider,
        "model_id": model_id,
        "quantization": quantization,
    }

    provider_dd = widgets.Dropdown(
        options=list(model_options.keys()),
        value=provider,
        description="Provider:",
    )
    model_dd = widgets.Dropdown(description="Model:")

    def update_model_options(*_):
        selection["provider"] = provider_dd.value
        model_dd.options = [
            (f"{option_model_id} | quantization={option_quantization}", (option_model_id, option_quantization))
            for option_model_id, option_quantization in model_options[selection["provider"]]
        ]
        model_dd.value = model_dd.options[0][1]

    def update_model_value(change):
        selection["model_id"], selection["quantization"] = change["new"]

    provider_dd.observe(update_model_options, names="value")
    model_dd.observe(update_model_value, names="value")

    update_model_options()
    return provider_dd, model_dd, selection


def resume_from_previous_results_folder(
        provider: str,
        model_id: str,
        resume_from: str,
        resume_to: str,
     ) -> str | None:

    if not os.path.isdir(resume_from):
        return None

    model_id_simplified = model_id.replace("/", "_")
    base_experiment_name = f"benchmark_{provider}_{model_id_simplified}_"

    for filename in os.listdir(resume_from):
        candidate_file_path = os.path.join(resume_from, filename)
        if filename.startswith(base_experiment_name) and os.path.isdir(candidate_file_path):
            dest_folder = os.path.join(resume_to, filename)
            if not os.path.exists(dest_folder):
                shutil.copytree(candidate_file_path, dest_folder)
            return filename

    return None


def zip_results_for_export(execution_env: str, summary_path: str) -> str | None:
    if execution_env not in ("colab", "kaggle"):
        return None

    benchmark_result_dir = os.path.dirname(summary_path)
    benchmark_name = os.path.basename(benchmark_result_dir)
    zip_path = os.path.join(os.path.dirname(benchmark_result_dir), f"{benchmark_name}_results_zip")
    shutil.make_archive(zip_path, "zip", benchmark_result_dir)
    return f"{zip_path}.zip"
