import argparse
import os
import warnings
import datetime
import shutil
from typing import Any

from langchain.chat_models.base import BaseChatModel

import experiments_util
from experiments_util import create_experiment_config, run_and_save_experiments
from chat_model_builder import build_chat_model


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")


def _make_results_folder_name(provider: str, model_id: str, quantization: str | None) -> str:
    model_id_simplified = model_id.replace("/", "_")
    return f"benchmark_{provider}_{model_id_simplified}_" + (f"quant{quantization}_" if quantization else "")


def find_previous_results_folder(
        provider: str,
        model_id: str,
        quantization: str | None,
        resume_from: str,
        resume_to: str,
     ) -> str | None:

    if not os.path.isdir(resume_from):
        return None

    base_experiment_name = _make_results_folder_name(provider, model_id, quantization)

    for filename in os.listdir(resume_from):
        candidate_file_path = os.path.join(resume_from, filename)
        if filename.startswith(base_experiment_name) and os.path.isdir(candidate_file_path):
            dest_folder = os.path.join(resume_to, filename)
            if not os.path.exists(dest_folder):
                shutil.copytree(candidate_file_path, dest_folder)
            return filename

    return None


# Each item is a configuration (global observation?, show numbers AND cells separator?, history size).
# Index in this list is the stable identifier used by `config_indices` in run_benchmark_minigrid,
# and by the configuration-selector widget in run_full_benchmark_minigrid.ipynb.
DEFAULT_CONFIG_PARAMS: list[tuple[bool, bool, int]] = [
    (True, False, 1),
    (True, True, 1),
    (True, False, 5),
    (True, True, 5),
    (False, False, 1),
    (False, True, 1),
    (False, False, 5),
    (False, True, 5),
]


def _config_label(global_view: bool, show_numbers_and_separators: bool, history_size: int) -> str:
    view = "global" if global_view else "local"
    fmt = "annotated" if show_numbers_and_separators else "simple"
    return f"{view} view | {fmt} format | history={history_size}"


CONFIG_LABELS: list[str] = [_config_label(*params) for params in DEFAULT_CONFIG_PARAMS]


def _build_configs(model_name: str, model: Any, config_indices: list[int] | None = None) -> list[dict[str, Any]]:
    """
    Builds ReAct agent configurations for a subset (or all, by default) of the 8
    standard (view x format x history) combinations in DEFAULT_CONFIG_PARAMS.
    """
    selected_params = (
        DEFAULT_CONFIG_PARAMS if config_indices is None
        else [DEFAULT_CONFIG_PARAMS[i] for i in config_indices]
    )

    experiment_configs = [
        create_experiment_config(model_name, model, global_view=gv, show_numbers=num_and_sep, separate_cells=num_and_sep, history_size=hist_sz,)
        for (gv, num_and_sep, hist_sz) in selected_params
    ]

    return experiment_configs


def run_benchmark_minigrid(
        provider: str,
        model_id: str,
        api_key: str | None = None,
        results_base_dir: str | None = None,
        results_folder_name: str | None = None,
        max_new_tokens: int = 2048,
        quantization: str | None = None,
        config_indices: list[int] | None = None,
        verbose: bool = True,
    ):
    """
    Executa benchmark MiniGrid para um modelo, usando 8 configuracoes fixas por padrao
    (ou um subconjunto delas, se `config_indices` for informado).

    Configuracoes disponiveis (8, ver DEFAULT_CONFIG_PARAMS / CONFIG_LABELS), variando
    estas 3 características:
    - Visão presente na observação      : global x local
    - Formato da observação             : simples x especial (com números e separadores)
    - Tamanho do histórico de mensagens : 1 x 5 últimas mensagens

    config_indices: indices (0-7) de DEFAULT_CONFIG_PARAMS/CONFIG_LABELS a executar.
        Se None (padrao), executa todas as 8 configuracoes.
    """
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Provider invalido: {provider}. Valores aceitos: {SUPPORTED_PROVIDERS}")

    if quantization is not None and quantization != "none" and provider != "hf":
        warnings.warn("Quantization is not supported for providers other than HuggingFace. Ignoring quantization parameter.")
    quantization = quantization if provider == "hf" else None

    if results_base_dir:
        experiments_util.RESULTS_BASE_DIR = os.path.abspath(results_base_dir)

    model = build_chat_model(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        max_output_tokens=max_new_tokens,
        hf_quantization=quantization
    )
    configs = _build_configs(model_name=model_id, model=model, config_indices=config_indices)

    if results_folder_name is None or results_folder_name.strip() == "":
        results_folder_name = _make_results_folder_name(provider, model_id, quantization)
        curr_date_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mmin")
        results_folder_name = f"{results_folder_name}{curr_date_time_str}"

    run_results = run_and_save_experiments(configs, experiment_name=results_folder_name, verbose=verbose)

    return run_results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa benchmark MiniGrid (8 configuracoes) para um modelo especifico."
    )
    parser.add_argument("provider", choices=SUPPORTED_PROVIDERS, help="Um desses: openai | deepseek | hf")
    parser.add_argument("model_id", help="ID do modelo (Exemplos: gpt-5.4-mini, deepseek-v4-flash, google/gemma-3-4b-it)")
    parser.add_argument("--results-dir", default=None, help="Diretorio base para salvar resultados")
    parser.add_argument("--api-key", default=None, help="API key (opcional). Se omitido, busca nas variaveis de ambiente")
    parser.add_argument("--quiet", action="store_true", help="Desativa barras de progresso")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Numero maximo de tokens gerados pelo modelo por resposta")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    final_results, filepath = run_benchmark_minigrid(
        provider=args.provider,
        model_id=args.model_id,
        results_base_dir=args.results_dir,
        api_key=args.api_key,
        verbose=not args.quiet,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Benchmark concluido. Configuracoes executadas: {len(final_results)}")
    print(f"Arquivo de resumo: {filepath}")


if __name__ == "__main__":
    main()
