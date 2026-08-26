"""Command-line entrypoint coverage."""

from ifrs9_ecl.cli import build_parser


def test_parser_exposes_every_reproducible_pipeline_phase() -> None:
    parser = build_parser()

    for command in ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "run-all"):
        args = parser.parse_args([command])
        assert callable(args.handler)
        assert args.config


def test_pipeline_commands_accept_a_local_data_root_override() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["phase5", "--config", "config/project.yaml", "--data-root", "/tmp/freddie"]
    )

    assert args.config == "config/project.yaml"
    assert args.data_root == "/tmp/freddie"
