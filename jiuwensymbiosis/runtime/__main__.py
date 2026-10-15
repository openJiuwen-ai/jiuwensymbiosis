"""Operator inspection and explicit reconciliation of stale resource records."""

import argparse
import json

from jiuwensymbiosis.runtime import CleanupReport, ResourceManager


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect local resource reservations; never connects hardware.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("resources", help="show resource records")
    reconcile = commands.add_parser("reconcile", help="clear a dead owner's records after external cleanup")
    reconcile.add_argument("operation_id")
    reconcile.add_argument(
        "--cleanup-confirmed",
        action="store_true",
        help="attest that in-flight work ended and devices/sidecars were cleaned up",
    )
    args = parser.parse_args(argv)
    manager = ResourceManager()
    if args.command == "resources":
        print(json.dumps(manager.records(), ensure_ascii=False, indent=2))
        return 0
    if not args.cleanup_confirmed:
        parser.error("inspect resources and finish external cleanup before passing --cleanup-confirmed")
    try:
        manager.reconcile(args.operation_id, CleanupReport())
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Reconciliation refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
