#!/usr/bin/env python3
"""
Main entry point for phetk command-line interface.
Provides subcommands for different phetk modules.
"""

import sys
import argparse

# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk.phewas import main as phewas_main
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk.demo import run as demo_run
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk.cohort import main_by_genotype, main_add_covariates
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk.phecode import main_count_phecode, main_add_age_at_first_event, main_add_phecode_time_to_event


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
    
    # Add cohort subcommand with nested subcommands
    cohort_parser = subparsers.add_parser(
        'cohort',
        help='Cohort generation and covariate management'
    )
    cohort_subparsers = cohort_parser.add_subparsers(
        dest='cohort_command',
        help='Cohort subcommands',
        required=True
    )
    
    # Add cohort by-genotype subcommand
    by_genotype_parser = cohort_subparsers.add_parser(
        'by-genotype',
        help='Generate cohort based on genotype of variant of interest',
        add_help=False  # Let the cohort module handle its own help
    )
    
    # Add cohort add-covariates subcommand
    add_covariates_parser = cohort_subparsers.add_parser(
        'add-covariates', 
        help='Add demographic, clinical, and genetic covariates to existing cohort',
        add_help=False  # Let the cohort module handle its own help
    )
    
    # Add phecode subcommand with nested subcommands
    phecode_parser = subparsers.add_parser(
        'phecode',
        help='ICD code extraction and phecode mapping'
    )
    phecode_subparsers = phecode_parser.add_subparsers(
        dest='phecode_command',
        help='Phecode subcommands',
        required=True
    )
    
    # Add phecode count-phecode subcommand
    count_phecode_parser = phecode_subparsers.add_parser(
        'count-phecode',
        help='Generate phecode counts from ICD code data',
        add_help=False  # Let the phecode module handle its own help
    )
    
    # Add phecode add-age-at-first-event subcommand
    add_age_parser = phecode_subparsers.add_parser(
        'add-age-at-first-event',
        help='Calculate age at first phecode event for each participant',
        add_help=False  # Let the phecode module handle its own help
    )
    
    # Add phecode add-phecode-time-to-event subcommand
    add_time_parser = phecode_subparsers.add_parser(
        'add-phecode-time-to-event',
        help='Calculate time from study start to first phecode event for survival analysis',
        add_help=False  # Let the phecode module handle its own help
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
    elif args.command == 'cohort':
        if args.cohort_command == 'by-genotype':
            # Pass remaining arguments to cohort by-genotype main
            sys.argv = ['phetk-cohort-by-genotype'] + remaining_args
            main_by_genotype()
        elif args.cohort_command == 'add-covariates':
            # Pass remaining arguments to cohort add-covariates main
            sys.argv = ['phetk-cohort-add-covariates'] + remaining_args
            main_add_covariates()
        else:
            cohort_parser.print_help()
            sys.exit(1)
    elif args.command == 'phecode':
        if args.phecode_command == 'count-phecode':
            # Pass remaining arguments to phecode count-phecode main
            sys.argv = ['phetk-phecode-count-phecode'] + remaining_args
            main_count_phecode()
        elif args.phecode_command == 'add-age-at-first-event':
            # Pass remaining arguments to phecode add-age-at-first-event main
            sys.argv = ['phetk-phecode-add-age-at-first-event'] + remaining_args
            main_add_age_at_first_event()
        elif args.phecode_command == 'add-phecode-time-to-event':
            # Pass remaining arguments to phecode add-phecode-time-to-event main
            sys.argv = ['phetk-phecode-add-phecode-time-to-event'] + remaining_args
            main_add_phecode_time_to_event()
        else:
            phecode_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()