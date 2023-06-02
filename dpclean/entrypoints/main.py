import argparse
import json
from typing import List, Optional

from dpclean.flow import build_workflow


def main_parser():
    parser = argparse.ArgumentParser(
        description="Data cleaning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Valid subcommands", dest="command")

    parser_submit = subparsers.add_parser(
        "submit",
        help="Submit a data-cleaning workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_submit.add_argument("CONFIG", help="the config file.")
    return parser


def parse_args(args: Optional[List[str]] = None):
    """Commandline options argument parsing.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option
        None takes arguments from sys.argv
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    if args.command == "submit":
        with open(args.CONFIG, "r") as f:
            config = json.load(f)
        wf = build_workflow(config)
        wf.submit()


if __name__ == "__main__":
    main()
