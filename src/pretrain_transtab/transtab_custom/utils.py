import json
from pathlib import Path


def unwrap_dataset(ds):
    """
    TransTab load_data 결과가 (X, y) 일 수도 있고 [(X, y)] 일 수도 있어서 정리
    """
    if isinstance(ds, (list, tuple)) and len(ds) == 1:
        inner = ds[0]
        if isinstance(inner, (list, tuple)) and len(inner) == 2:
            return inner[0], inner[1]
    if isinstance(ds, (list, tuple)) and len(ds) == 2:
        return ds[0], ds[1]
    raise ValueError(f"Unexpected dataset format: type={type(ds)}, repr={repr(ds)[:300]}")

def save_column_info(
    run_dir: Path,
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
) -> None:
    info = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "binary_columns": binary_columns,
        "num_categorical": len(categorical_columns),
        "num_numerical": len(numerical_columns),
        "num_binary": len(binary_columns),
    }

    with open(run_dir / "column_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)