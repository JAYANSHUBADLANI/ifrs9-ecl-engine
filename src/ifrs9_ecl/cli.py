"""Command line interface for the IFRS9 ECL research workflow."""

from __future__ import annotations

import argparse
import sys

from ifrs9_ecl.config import DEFAULT_CONFIG, load_config
from ifrs9_ecl.phase0 import run_phase0, summary_as_text
from ifrs9_ecl.phase1 import run_phase1
from ifrs9_ecl.phase2 import run_phase2
from ifrs9_ecl.phase3 import run_phase3
from ifrs9_ecl.phase4 import run_phase4
from ifrs9_ecl.phase5 import run_phase5


def _run_phase0_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    summary = run_phase0(
        config,
        vintage=args.vintage,
        maximum_loans=args.maximum_loans,
    )
    print(summary_as_text(summary))


def _run_phase1_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    summary = run_phase1(config)
    print(summary_as_text(summary))


def _run_phase2_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    print(summary_as_text(run_phase2(config)))


def _run_phase3_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    print(summary_as_text(run_phase3(config)))


def _run_phase4_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    print(summary_as_text(run_phase4(config)))


def _run_phase5_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    print(summary_as_text(run_phase5(config)))


def _run_all_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, data_root=args.data_root)
    summaries = {
        "phase1": run_phase1(config),
        "phase2": run_phase2(config),
        "phase3": run_phase3(config),
        "phase4": run_phase4(config),
        "phase5": run_phase5(config),
    }
    print(summary_as_text(summaries))


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ifrs9-ecl",
        description="IFRS9-style expected credit loss research engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0 = subparsers.add_parser(
        "phase0", description="Run the real-data panel and transition smoke test"
    )
    phase0.add_argument("--config", default=str(DEFAULT_CONFIG))
    phase0.add_argument("--data-root")
    phase0.add_argument("--vintage")
    phase0.add_argument("--maximum-loans", type=int)
    phase0.set_defaults(handler=_run_phase0_command)

    phase1 = subparsers.add_parser(
        "phase1", description="Build the sampled multi-vintage panel and roll rates"
    )
    _add_config_arguments(phase1)
    phase1.set_defaults(handler=_run_phase1_command)

    phase2 = subparsers.add_parser(
        "phase2", description="Fit the monthly default hazard and lifetime PD curves"
    )
    _add_config_arguments(phase2)
    phase2.set_defaults(handler=_run_phase2_command)

    phase3 = subparsers.add_parser(
        "phase3", description="Fit cure, conditional LGD, and amortizing EAD"
    )
    _add_config_arguments(phase3)
    phase3.set_defaults(handler=_run_phase3_command)

    phase4 = subparsers.add_parser(
        "phase4", description="Assign stages and calculate discounted scenario ECL"
    )
    _add_config_arguments(phase4)
    phase4.set_defaults(handler=_run_phase4_command)

    phase5 = subparsers.add_parser(
        "phase5", description="Backtest snapshot ECL against later Actual Loss"
    )
    _add_config_arguments(phase5)
    phase5.set_defaults(handler=_run_phase5_command)

    run_all = subparsers.add_parser(
        "run-all", description="Run or reuse every phase from panel through backtest"
    )
    _add_config_arguments(run_all)
    run_all.set_defaults(handler=_run_all_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
