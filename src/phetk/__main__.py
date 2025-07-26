#!/usr/bin/env python3
"""
Main entry point for phetk command-line interface.
Provides subcommands for different phetk modules.
"""

import sys
import argparse
from phetk.phewas import main as phewas_main
from phetk.demo import run as demo_run


def main():
    """Main entry point for phetk CLI with subcommands."""
    parser = argparse.ArgumentParser(
        prog='phetk',
        description='The Phenotype Toolkit (PheTK) - Command Line Interface'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Add phewas subcommand
    phewas_parser = subparsers.add_parser(
        'phewas',
        help='Run PheWAS analysis',
        add_help=False  # Let the phewas module handle its own help
    )
    
    # Add demo subcommand
    demo_parser = subparsers.add_parser(
        'demo',
        help='Run PheTK demonstration'
    )
    
    # Parse only the command, not the full arguments
    args, remaining_args = parser.parse_known_args()
    
    # Route to appropriate function
    if args.command == 'phewas':
        # Pass remaining arguments to phewas main
        sys.argv = ['phetk-phewas'] + remaining_args
        phewas_main()
    elif args.command == 'demo':
        demo_run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()