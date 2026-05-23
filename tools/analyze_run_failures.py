"""
Analisa falhas de API nos resultados de experimentos.

Detecta runs em que o agente não completou nenhum passo, indicando que a
primeira chamada ao modelo falhou antes de qualquer resposta ser registrada.

Uso:
    python tools/analyze_api_failures.py                     # usa pasta results/ padrão
    python tools/analyze_api_failures.py caminho/para/results
"""

import json
import os
import sys
from pathlib import Path


def is_api_failure(run_data: dict) -> bool:
    """
    Um run é considerado falha de API quando:
    - steps == 0 (nenhum passo executado), E
    - o histórico contém apenas mensagens do tipo 'human' (a IA nunca respondeu).
    """
    if int(run_data.get("steps", -1)) != 0:
        return False
    history = run_data.get("history", [])
    if not history:
        return False
    return all(msg.get("role") == "human" for msg in history)


def scan_results(results_dir: Path) -> list[dict]:
    failures = []
    for run_file in sorted(results_dir.rglob("*.json")):
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Pula arquivos de resumo: são dicts sem a chave "experiment" de run individual
        if not isinstance(data, dict) or "run" not in data:
            continue
        if is_api_failure(data):
            failures.append({
                "file": str(run_file.relative_to(results_dir)),
                "experiment": data.get("experiment", "?"),
                "env": data.get("env", "?"),
                "run": data.get("run", "?"),
                "seed": data.get("seed", "?"),
                "history_len": len(data.get("history", [])),
            })
    return failures


def print_report(failures: list[dict], results_dir: Path) -> None:
    if not failures:
        print("Nenhuma falha de API detectada.")
        return

    # Agrupa por experimento
    by_experiment: dict[str, list[dict]] = {}
    for f in failures:
        by_experiment.setdefault(f["experiment"], []).append(f)

    print(f"{'='*60}")
    print(f"Falhas de API detectadas: {len(failures)} runs")
    print(f"Diretório analisado: {results_dir}")
    print(f"{'='*60}\n")

    for experiment, runs in sorted(by_experiment.items()):
        print(f"  Experimento: {experiment}  ({len(runs)} falhas)")
        for r in runs:
            print(f"    run {r['run']:>2} | env: {r['env']} | seed: {r['seed']} | history_len: {r['history_len']} | {r['file']}")
        print()

    # Resumo por experimento
    print(f"{'─'*60}")
    print(f"{'Experimento':<45} {'Falhas':>6}")
    print(f"{'─'*60}")
    for experiment, runs in sorted(by_experiment.items()):
        print(f"{experiment:<45} {len(runs):>6}")
    print(f"{'─'*60}")
    print(f"{'TOTAL':<45} {len(failures):>6}")


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Localiza a pasta results/ relativa a este script (tools/ -> raiz -> results/)
        results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print(f"Erro: diretório não encontrado: {results_dir}", file=sys.stderr)
        sys.exit(1)

    failures = scan_results(results_dir)
    print_report(failures, results_dir)


if __name__ == "__main__":
    main()
