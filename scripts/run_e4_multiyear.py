"""Run the E4 temporal generalization pipeline across multiple years."""

import argparse
import subprocess
import sys
from satellite_health.data.paths import load_data_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2013)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")

    years = range(args.start_year, args.end_year + 1)

    print("E4 multiyear pipeline")
    print("Years:", list(years))
    print("Python:", sys.executable)

    paths = load_data_paths("configs/data/paths.local.yaml")

    interim_dir = paths["interim"] / "mission1" / "group8"
    comparison_dir = (
        paths["outputs"] / "experiments" / "e4" / "comparisons"
    )

    for year in years:
        print(f"\n===== E4 YEAR {year} =====", flush=True)

        telemetry_file = interim_dir / f"telemetry_{year}.pkl"
        windows_file = interim_dir / f"windows_{year}.csv"
        comparison_file = comparison_dir / f"comparison_{year}.csv"

        if comparison_file.exists():
            print("Comparison already exists; skipping year.", flush=True)
            continue

        commands = []

        if not telemetry_file.exists():
            commands.append([
                sys.executable,
                "scripts/prepare_group8_year.py",
                "--year",
                str(year),
            ])

        if not windows_file.exists():
            commands.append([
                sys.executable,
                "scripts/build_year_window_table.py",
                "--year",
                str(year),
            ])

        commands.append([
            sys.executable,
            "scripts/run_e4_comparison.py",
            "--year",
            str(year),
        ])

        for command in commands:
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)

        print(f"Completed year {year}", flush=True)

if __name__ == "__main__":
    main()
