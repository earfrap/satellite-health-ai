from pathlib import Path

import yaml

from satellite_health.data.paths import load_data_paths


def test_load_data_paths(tmp_path: Path) -> None:
    """Data paths should be constructed correctly from configuration."""

    data_root = tmp_path / "satellite-health-ai-data"

    for directory in ("raw", "interim", "processed", "outputs"):
        (data_root / directory).mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "paths.local.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"data_root": str(data_root)}, file)

    paths = load_data_paths(config_path)

    assert paths["root"] == data_root
    assert paths["raw"] == data_root / "raw"
    assert paths["interim"] == data_root / "interim"
    assert paths["processed"] == data_root / "processed"
    assert paths["outputs"] == data_root / "outputs"
