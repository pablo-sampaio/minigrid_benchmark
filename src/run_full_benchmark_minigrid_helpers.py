import os
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
import warnings


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf", "qwen-local-server")

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
        ("google/gemma-3-4b-it", None),
        ("google/gemma-3-12b-it", "4bit"),
        ("google/gemma-4-E2B-it", None),
        ("google/gemma-4-E4B-it", None),
        ("google/gemma-4-12B-it", "4bit"),
        ("Qwen/Qwen2.5-3B-Instruct", None),
        ("Qwen/Qwen2.5-7B-Instruct", "8bit"),
        ("Qwen/Qwen3-4B", None),  # mais antigo, de propósito híbrido (instrutivo + thinking)
        ("Qwen/Qwen3-4B-Instruct-2507", None),
        ("Qwen/Qwen3-4B-Thinking-2507", None),
        #("Qwen/Qwen3-4B-Instruct-2507-FP8", None),
        #("Qwen/Qwen3-4B-Thinking-2507-FP8", None),
        ("Qwen/Qwen3.5-0.8B", None),
        ("Qwen/Qwen3.5-2B", None),
        ("Qwen/Qwen3.5-4B", None),
        ("WeiboAI/VibeThinker-3B", "8bit"),
    ],
    "qwen-local-server": [
        ("Qwen/Qwen3.5-0.8B", None),
        ("Qwen/Qwen3.5-2B", None),
        ("Qwen/Qwen3.5-4B", None),
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
    if provider == "qwen-local-server":
        secret_names = ["QWEN_LOCAL_API_KEY", "LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY"]

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

    if provider == "qwen-local-server":
        return "EMPTY"

    warnings.warn(f"API key not found for provider '{provider}'.")
    return None


def ensure_qwen_local_server(
        provider: str,
        model_id: str,
        execution_env: str,
        repo_path: str,
        port: int = 8000,
        startup_timeout_s: int = 240,
        max_model_len: int = 32768,
    ) -> dict[str, str | int | None]:
    provider = provider.strip().lower()
    if provider != "qwen-local-server":
        return {
            "started": False,
            "base_url": None,
            "log_path": None,
            "pid": None,
        }

    base_url = f"http://127.0.0.1:{port}/v1"
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    if _is_server_ready(base_url):
        return {
            "started": False,
            "base_url": base_url,
            "log_path": None,
            "pid": None,
        }

    if execution_env in ("colab", "kaggle"):
        _install_vllm_if_missing()

    logs_dir = os.path.join(repo_path, "results", "server_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"qwen_local_server_{model_id.replace('/', '_')}_{int(time.time())}.log")

    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--language-model-only",
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=repo_path,
        )

    _wait_for_server_ready(base_url=base_url, timeout_s=startup_timeout_s)
    return {
        "started": True,
        "base_url": base_url,
        "log_path": log_path,
        "pid": process.pid,
    }


def _install_vllm_if_missing() -> None:
    try:
        import vllm  # noqa: F401

        return
    except Exception:
        pass

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "vllm",
        "--torch-backend=auto",
        "--extra-index-url",
        "https://wheels.vllm.ai/nightly",
    ]
    subprocess.check_call(command)


def _is_server_ready(base_url: str) -> bool:
    models_url = f"{base_url}/models"
    request = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
            return 200 <= response.status < 300
    except Exception:
        return False


def _wait_for_server_ready(base_url: str, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if _is_server_ready(base_url):
            return
        time.sleep(2)

    raise TimeoutError(
        f"Qwen local server did not become ready at '{base_url}' within {timeout_s} seconds."
    )


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
