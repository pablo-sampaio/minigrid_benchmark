from __future__ import annotations

import argparse
from pathlib import Path


def rename_one_digit_json_files(root: Path, dry_run: bool = False) -> tuple[int, int]:
	"""
	Recursively rename files matching <digit>.json to 0<digit>.json.

	Example: 1.json -> 01.json

	Returns:
		(renamed_count, skipped_count)
	"""
	renamed_count = 0
	skipped_count = 0

	for file_path in root.rglob("*.json"):
		if not file_path.is_file():
			continue

		stem = file_path.stem
		if len(stem) == 1 and stem.isdigit():
			new_name = f"0{stem}.json"
			target_path = file_path.with_name(new_name)

			if target_path.exists():
				print(f"SKIP (target exists): {file_path} -> {target_path}")
				skipped_count += 1
				continue

			print(f"RENAME: {file_path} -> {target_path}")
			if not dry_run:
				file_path.rename(target_path)
			renamed_count += 1

	return renamed_count, skipped_count


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Recursively rename <digit>.json files to two-digit format (e.g., 1.json -> 01.json)."
	)
	parser.add_argument(
		"root",
		nargs="?",
		default="results",
		help="Root folder to scan recursively (default: results).",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show what would be renamed without changing files.",
	)

	args = parser.parse_args()
	root = Path(args.root)

	if not root.exists() or not root.is_dir():
		raise SystemExit(f"Invalid directory: {root}")

	renamed_count, skipped_count = rename_one_digit_json_files(root, dry_run=args.dry_run)

	mode = "DRY-RUN" if args.dry_run else "DONE"
	print(f"{mode}: renamed={renamed_count}, skipped={skipped_count}")


if __name__ == "__main__":
	main()
