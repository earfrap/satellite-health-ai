from pathlib import Path

import yaml


def load_data_paths(config_path: str | Path) -> dict[str, Path]:
    """Load and validate local data storage paths."""

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_root = Path(config["data_root"]).expanduser()

    paths = {
        "root": data_root,
        "raw": data_root / "raw",
        "interim": data_root / "interim",
        "processed": data_root / "processed",
        "outputs": data_root / "outputs",
    }

    missing = [path for path in paths.values() if not path.exists()]

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required data directories:\n{missing_text}"
        )

    return paths
