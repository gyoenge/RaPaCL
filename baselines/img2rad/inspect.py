import os
import glob
import argparse
import pandas as pd
import yaml


# ------------------------------------------------------------------
# utils
# ------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# core
# ------------------------------------------------------------------
def inspect_parquet_first_rows(
    radiomics_parquet_dir: str,
    max_files: int | None = None,
    show_columns: bool = False,
    check_nan: bool = True,
) -> None:
    parquet_files = sorted(
        glob.glob(os.path.join(radiomics_parquet_dir, "*.parquet"))
    )

    if max_files:
        parquet_files = parquet_files[:max_files]

    print(f"📁 radiomics_dir: {radiomics_parquet_dir}")
    print(f"Found {len(parquet_files)} parquet files\n")

    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(parquet_path)

            print("=" * 80)
            print(f"[FILE] {os.path.basename(parquet_path)}")
            print(f"shape: {df.shape}")

            if df.empty:
                print("Empty dataframe")
                continue

            first_row = df.iloc[0]

            print("\n[First Row]")
            print(first_row)

            if show_columns:
                print("\n[Columns]")
                print(df.columns.tolist())

            if check_nan:
                nan_count = first_row.isna().sum()
                print(f"\n[NaN count] {nan_count}")

        except Exception as e:
            print(f"Error loading {parquet_path}: {e}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inspect utilities")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["parquet"])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--show_columns", action="store_true")
    parser.add_argument("--no_nan_check", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml(args.config)

    # 👉 여기 핵심
    radiomics_dir = cfg["data"]["radiomics_parquet_dir"]

    if args.mode == "parquet":
        inspect_parquet_first_rows(
            radiomics_parquet_dir=radiomics_dir,
            max_files=args.max_files,
            show_columns=args.show_columns,
            check_nan=not args.no_nan_check,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()