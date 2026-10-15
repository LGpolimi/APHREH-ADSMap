"""APHREH-ADSMap 2.0.1 desktop launcher and optional headless entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def configure_numerical_threads() -> None:
    # Set limits before importing NumPy/SciPy or launching child workers.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def run_headless(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run an APHREH-ADSMap project without the GUI")
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration file")
    parser.add_argument("--input", type=Path, help="Override source-data directory")
    parser.add_argument("--output", type=Path, help="Override project/output directory")
    parser.add_argument("--phase", choices=("all", "reference", "calibration", "robustness", "confounder"), default="all")
    args = parser.parse_args(argv)
    try:
        from aphreh_adsmap.config import load_config, validate
        from aphreh_adsmap.workflow import run_project
        config = load_config(args.config)
        if args.input is not None:
            config["data_dir"] = str(args.input.expanduser().resolve())
        if args.output is not None:
            config["output_dir"] = str(args.output.expanduser().resolve())
        validate(config)
        run_project(config, phase=args.phase)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed checkpoint records remain on disk.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print("Run 'python main.py' to open the GUI, or 'python main.py --headless --help' for batch mode.")
        return 0
    configure_numerical_threads()
    if "--headless" in args:
        args.remove("--headless")
        return run_headless(args)
    if args:
        print("Unexpected arguments for desktop mode. Use --headless for batch execution.", file=sys.stderr)
        return 2
    from aphreh_adsmap.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
