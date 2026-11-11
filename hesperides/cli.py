"""Command-line orchestration for the Hesperides package."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Sequence, TextIO

from hesperides.api import get_price_binomial_european


class BooleanFlagParser:
    """Parse the legacy true/false values accepted by --call."""

    TRUE_VALUES = ("true", "1", "yes")

    def parse(self, value: str) -> bool:
        """Return True only for the historical truthy spellings."""
        return str(value).lower() in self.TRUE_VALUES


@dataclass(frozen=True)
class BinomialEuropeanPriceCommand:
    """Execute the binomial European pricing CLI command."""

    def execute(self, args: argparse.Namespace, stderr: TextIO) -> int:
        """Run the command and return a process exit code."""
        try:
            if args.call is not None:
                is_call = args.call
            elif args.put:
                is_call = False
            else:
                print(
                    "Error: must specify either --call true/false or --put",
                    file=stderr,
                )
                return 1

            price = get_price_binomial_european(
                St=args.St,
                K=args.K,
                T=args.T,
                R=args.R,
                u=args.u,
                d=args.d,
                call=is_call,
            )
            print(f"{price:.10f}")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=stderr)
            return 1


@dataclass(frozen=True)
class HesperidesCli:
    """Build and execute the command-line interface."""

    bool_parser: BooleanFlagParser = field(default_factory=BooleanFlagParser)
    binomial_command: BinomialEuropeanPriceCommand = field(
        default_factory=BinomialEuropeanPriceCommand
    )

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        return cls().run(argv=argv)

    def build_parser(self) -> argparse.ArgumentParser:
        """Create the CLI parser."""
        parser = argparse.ArgumentParser(
            description="Price a European option using the binomial model."
        )
        subparsers = parser.add_subparsers(dest="command")
        price_parser = subparsers.add_parser("price-binomial-european")

        price_parser.add_argument(
            "--St",
            type=float,
            required=True,
            help="Current spot price",
        )
        price_parser.add_argument("--K", type=float, required=True, help="Strike price")
        price_parser.add_argument(
            "--T",
            type=int,
            required=True,
            help="Steps to maturity",
        )
        price_parser.add_argument("--R", type=float, required=True, help="Risk-free rate")
        price_parser.add_argument("--u", type=float, required=True, help="Up factor")
        price_parser.add_argument("--d", type=float, required=True, help="Down factor")
        price_parser.add_argument(
            "--call",
            type=self.bool_parser.parse,
            default=None,
            help="Price a call option (true/false)",
        )
        price_parser.add_argument(
            "--put",
            action="store_true",
            help="Price a put option (alternative to --call false)",
        )
        return parser

    def run(
        self,
        argv: Sequence[str] | None = None,
        stderr: TextIO = sys.stderr,
    ) -> int:
        """Execute the CLI and return a process exit code."""
        parser = self.build_parser()
        args = parser.parse_args(argv)

        if args.command == "price-binomial-european":
            return self.binomial_command.execute(args=args, stderr=stderr)

        parser.print_help()
        return 0


main = HesperidesCli.main
