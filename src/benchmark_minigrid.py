import argparse
import os
import warnings
import datetime
from typing import Any

from langchain.chat_models.base import BaseChatModel

import experiments_util
from experiments_util import create_experiment_config, run_and_save_experiments
from chat_model_builder import resolve_api_key, build_chat_model


SUPPORTED_PROVIDERS = ("openai", "deepseek", "hf")


def _build_default_8_configs(model_name: str, model: Any) -> list[dict[str, Any]]:
    # each item is a configuration (gloval observation?, show numbers AND cells separator?, history size)
    config_params = [
        (True, False, 1),
        (True, True, 1), 
        (True, False, 5),
        (True, True, 5),
        (False, False, 1),
        (False, True, 1),
        (False, False, 5),
        (False, True, 5),
    ]

    experiment_configs = [ 
        create_experiment_config(model_name, model, global_view=gv, show_numbers=num_and_sep, separate_cells=num_and_sep, history_size=hist_sz,)
        for (gv, num_and_sep, hist_sz) in config_params
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
        verbose: bool = True,
    ):
    """
    Executa benchmark MiniGrid com 8 configuracoes fixas para um modelo.

    Configuracoes usadas (8), variando estas 3 características:
    - Visão presente na observação      : global x local
    - Formato da observação             : simples x especial (com números e separadores)
    - Tamanho do histórico de mensagens : 1 x 5 últimas mensagens
    """
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Provider invalido: {provider}. Valores aceitos: {SUPPORTED_PROVIDERS}")

    if quantization is not None and quantization != "none" and provider != "hf":
        warnings.warn("Quantization is not supported for providers other than HuggingFace. Ignoring quantization parameter.")
    quantization = quantization if provider == "hf" else None

    if results_base_dir:
        experiments_util.RESULTS_BASE_DIR = os.path.abspath(results_base_dir)

    resolved_key = resolve_api_key(provider, api_key)
    model = build_chat_model(
        provider=provider,
        model_id=model_id,
        api_key=resolved_key,
        max_output_tokens=max_new_tokens,
        hf_quantization=quantization
    )
    configs = _build_default_8_configs(model_name=model_id, model=model)

    if results_folder_name is None or results_folder_name.strip() == "":
        model_id_simplified = model_id.replace("/", "_")  # replace / with _ for file/folder names
        curr_date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"benchmark_{provider}_{model_id_simplified}_{curr_date_time_str}"

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
    parser.add_argument("--hf-max-new-tokens", type=int, default=2048, help="max_new_tokens para provider hf (HuggingFace)")
    parser.add_argument(
        "--hf-no-sampling",
        action="store_true",
        help="Desativa do_sample no pipeline HuggingFace",
    )
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
        hf_max_new_tokens=args.hf_max_new_tokens,
        hf_do_sample=not args.hf_no_sampling,
    )

    print(f"Benchmark concluido. Configuracoes executadas: {len(final_results)}")
    print(f"Arquivo de resumo: {filepath}")


if __name__ == "__main__":
    main()
