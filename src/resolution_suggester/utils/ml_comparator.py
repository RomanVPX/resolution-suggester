# utils/ml_comparator.py
import argparse
import copy
import json
import logging
import os
import numpy as np
from typing import Dict, Tuple, Optional
from ..i18n import _

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..config import QualityMetrics, RICH_STYLES
from ..utils.cli import validate_paths
from ..core.image_analyzer import ImageAnalyzer


console = Console()


class MLComparator:
    """
    Utility to compare the results of image analysis with and without ML prediction.
    Runs the analysis twice and compares the results.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize with command line arguments."""
        self.original_args = args
        self.real_results_path = None
        self.ml_results_path = None
        self.comparison_results = {}

    def run_comparison(self) -> None:
        """Run the comparison process: two analyses and comparison."""
        # Force JSON output
        if not self.original_args.json_output:
            logging.info(_("Enabling JSON output for comparison"))
            self.original_args.json_output = True

        # Run without ML first
        real_args = copy.deepcopy(self.original_args)
        real_args.ml = False
        self.real_results_path = self._run_analysis(real_args, "real")

        # Run with ML
        ml_args = copy.deepcopy(self.original_args)
        ml_args.ml = True
        self.ml_results_path = self._run_analysis(ml_args, "ml")

        # Compare results
        self._compare_results()

        # Output comparison
        self._report_comparison()

    def _run_analysis(self, args: argparse.Namespace, run_type: str) -> Optional[str]:
        """Run analysis with given arguments and return the path to the results file."""
        logging.info(f"Starting {run_type} analysis run...")

        # Modify output path to make it unique
        # The original json path will be determined by the reporter
        original_paths = validate_paths(args.paths)

        # Create the analyzer and run it
        analyzer = ImageAnalyzer(args)
        reporters, output_paths = self._setup_reporters(args, run_type)

        try:
            # Process files
            for file_path in original_paths:
                try:
                    results, meta = analyzer.analyze_file(file_path)
                    if results:
                        for rep in reporters:
                            rep.write_results(os.path.basename(file_path), results, args.channels)
                except Exception as e:
                    logging.error(f"{_('Error processing file')} {file_path}: {e}")

            # Get the output JSON path
            json_path = output_paths.get('json')
            logging.info(f"{run_type.capitalize()} analysis complete. Results saved to: {json_path}")
            return json_path
        finally:
            # Close reporters
            for rep in reporters:
                try:
                    rep.__exit__(None, None, None)
                except Exception as e:
                    logging.error(f"Error closing reporter: {e}")

    def _setup_reporters(self, args: argparse.Namespace, run_type: str) -> Tuple[list, dict]:
        """Configure reporters with unique filenames for each run."""
        from ..utils.reporters import JSONReporter, get_json_log_filename

        reporters = []
        output_paths = {}

        # Get original JSON path
        json_path = get_json_log_filename(args)

        # Modify the path to include run type
        base, ext = os.path.splitext(json_path)
        modified_path = f"{base}_{run_type}{ext}"

        # Create and set up reporter
        json_reporter = JSONReporter(modified_path, QualityMetrics(args.metric))
        json_reporter.__enter__()
        reporters.append(json_reporter)
        output_paths['json'] = modified_path

        return reporters, output_paths

    def _compare_results(self) -> None:
        """Compare real and ML results and compute statistics."""
        if not self.real_results_path or not self.ml_results_path:
            logging.error("Missing results for comparison")
            return

        try:
            # Load results
            with open(self.real_results_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)

            with open(self.ml_results_path, 'r', encoding='utf-8') as f:
                ml_data = json.load(f)

            # Process each file
            for real_file_entry in real_data:
                file_name = real_file_entry.get('file')

                # Find matching ML entry
                ml_file_entry = next((item for item in ml_data if item.get('file') == file_name), None)

                if not ml_file_entry:
                    logging.warning(f"No matching ML results for file: {file_name}")
                    continue

                # Compare results for this file
                file_comparison = self._compare_file_results(real_file_entry, ml_file_entry)
                self.comparison_results[file_name] = file_comparison

            # Calculate overall statistics
            self._calculate_overall_statistics()

        except Exception as e:
            logging.error(f"Error comparing results: {e}")
            logging.debug("Details:", exc_info=True)

    def _compare_file_results(self, real_entry: Dict, ml_entry: Dict) -> Dict:
        """Compare real and ML results for a single file."""
        comparison = {
            'resolutions': {},
            'statistics': {
                'max_delta': 0.0,
                'min_delta': float('inf'),
                'median_delta': 0.0,
                'mean_delta': 0.0,
                'deltas': []
            }
        }

        real_results = real_entry.get('results', [])
        ml_results = ml_entry.get('results', [])

        # Match results by resolution
        for real_res in real_results:
            resolution = real_res.get('resolution')

            # Skip the original resolution (which has infinity values)
            if "original" in real_res.get('hint', '').lower():
                continue

            # Find matching ML result
            ml_res = next((r for r in ml_results if r.get('resolution') == resolution), None)

            if not ml_res:
                logging.warning(f"No matching ML result for resolution: {resolution}")
                continue

            # Compare values
            if self.original_args.channels:
                # Channel comparison
                real_channels = real_res.get('channels', {})
                ml_channels = ml_res.get('channels', {})

                channels_comparison = {}
                channel_deltas = []

                for channel, real_val in real_channels.items():
                    if channel in ml_channels:
                        # Handle "inf" strings
                        real_value = float('inf') if real_val == "inf" else float(real_val)
                        ml_value = float('inf') if ml_channels[channel] == "inf" else float(ml_channels[channel])

                        # Calculate absolute difference
                        if real_value == float('inf') and ml_value == float('inf'):
                            delta = 0.0
                        elif real_value == float('inf') or ml_value == float('inf'):
                            delta = float('inf')
                        else:
                            delta = abs(real_value - ml_value)

                        channels_comparison[channel] = {
                            'real': real_value,
                            'ml': ml_value,
                            'delta': delta
                        }

                        if delta != float('inf'):
                            channel_deltas.append(delta)

                # Calculate statistics for this resolution
                max_delta = max(channel_deltas) if channel_deltas else 0.0
                mean_delta = np.mean(channel_deltas) if channel_deltas else 0.0

                comparison['resolutions'][resolution] = {
                    'channels': channels_comparison,
                    'max_delta': max_delta,
                    'mean_delta': mean_delta
                }

                # Update overall statistics
                if max_delta > comparison['statistics']['max_delta']:
                    comparison['statistics']['max_delta'] = max_delta
                if max_delta < comparison['statistics']['min_delta']:
                    comparison['statistics']['min_delta'] = max_delta
                comparison['statistics']['deltas'].extend(channel_deltas)

            else:
                # Single value comparison
                real_value = float('inf') if real_res.get('value') == "inf" else float(real_res.get('value'))
                ml_value = float('inf') if ml_res.get('value') == "inf" else float(ml_res.get('value'))

                if real_value == float('inf') and ml_value == float('inf'):
                    delta = 0.0
                elif real_value == float('inf') or ml_value == float('inf'):
                    delta = float('inf')
                else:
                    delta = abs(real_value - ml_value)

                comparison['resolutions'][resolution] = {
                    'real': real_value,
                    'ml': ml_value,
                    'delta': delta
                }

                # Update overall statistics if delta is not infinity
                if delta != float('inf'):
                    if delta > comparison['statistics']['max_delta']:
                        comparison['statistics']['max_delta'] = delta
                    if delta < comparison['statistics']['min_delta']:
                        comparison['statistics']['min_delta'] = delta
                    comparison['statistics']['deltas'].append(delta)

        # Calculate median and mean
        deltas = comparison['statistics']['deltas']
        if deltas:
            comparison['statistics']['median_delta'] = np.median(deltas)
            comparison['statistics']['mean_delta'] = np.mean(deltas)
        else:
            comparison['statistics']['min_delta'] = 0.0

        return comparison

    def _calculate_overall_statistics(self) -> None:
        """Calculate overall statistics across all files."""
        self.overall_stats = {
            'max_median_delta_file': None,
            'min_median_delta_file': None,
            'max_median_delta': 0.0,
            'min_median_delta': float('inf'),
            'mean_median_delta': 0.0,
            'all_medians': []
        }

        for file_name, comparison in self.comparison_results.items():
            median_delta = comparison['statistics']['median_delta']

            self.overall_stats['all_medians'].append(median_delta)

            if median_delta > self.overall_stats['max_median_delta']:
                self.overall_stats['max_median_delta'] = median_delta
                self.overall_stats['max_median_delta_file'] = file_name

            if median_delta < self.overall_stats['min_median_delta']:
                self.overall_stats['min_median_delta'] = median_delta
                self.overall_stats['min_median_delta_file'] = file_name

        if self.overall_stats['all_medians']:
            self.overall_stats['mean_median_delta'] = np.mean(self.overall_stats['all_medians'])

    def _report_comparison(self) -> None:
        """Generate and display comparison report."""
        console.print()
        console.print(Text(_("ML vs Real Metrics Comparison"), style="bold cyan"))
        console.print()

        # Print per-file comparisons
        for file_name, comparison in self.comparison_results.items():
            self._print_file_comparison(file_name, comparison)

        # Print overall statistics
        self._print_overall_statistics()

    def _print_file_comparison(self, file_name: str, comparison: Dict) -> None:
        """Print comparison results for a single file."""
        console.print(Text(f"{_('File')}: {file_name} ({QualityMetrics(self.original_args.metric).upper()})", style="bold cyan"))

        # Create table
        table = Table(show_header=True, header_style="bold")

        # Add columns based on analysis type
        if self.original_args.channels:
            table.add_column(_("Resolution"), style="bold")
            table.add_column(_("Channel"))
            table.add_column(_("Real"))
            table.add_column(_("ML"))
            table.add_column(_("Delta"))
            table.add_column(_("Relative Error (%)"))
        else:
            table.add_column(_("Resolution"), style="bold")
            table.add_column(_("Real"))
            table.add_column(_("ML"))
            table.add_column(_("Delta"))
            table.add_column(_("Relative Error (%)"))

        # Add rows for each resolution
        for resolution, res_data in comparison['resolutions'].items():
            if self.original_args.channels:
                # Multi-channel display
                first_row = True
                channels_data = res_data.get('channels', {})

                for channel, ch_data in channels_data.items():
                    real_val = ch_data['real']
                    ml_val = ch_data['ml']
                    delta = ch_data['delta']

                    # Format values
                    real_str = "∞" if real_val == float('inf') else f"{real_val:.3f}"
                    ml_str = "∞" if ml_val == float('inf') else f"{ml_val:.3f}"
                    delta_str = "∞" if delta == float('inf') else f"{delta:.3f}"

                    # Calculate relative error
                    if real_val != 0 and real_val != float('inf') and delta != float('inf'):
                        rel_error = (delta / real_val) * 100
                        rel_error_str = f"{rel_error:.2f}%"
                    else:
                        rel_error_str = _("N/A")

                    # Determine row style based on delta
                    row_style = self._get_delta_style(delta)

                    if first_row:
                        table.add_row(
                            resolution, channel, real_str, ml_str, delta_str, rel_error_str,
                            style=row_style
                        )
                        first_row = False
                    else:
                        table.add_row(
                            "", channel, real_str, ml_str, delta_str, rel_error_str,
                            style=row_style
                        )
            else:
                # Single value display
                real_val = res_data['real']
                ml_val = res_data['ml']
                delta = res_data['delta']

                # Format values
                real_str = "∞" if real_val == float('inf') else f"{real_val:.3f}"
                ml_str = "∞" if ml_val == float('inf') else f"{ml_val:.3f}"
                delta_str = "∞" if delta == float('inf') else f"{delta:.3f}"

                # Calculate relative error
                if real_val != 0 and real_val != float('inf') and delta != float('inf'):
                    rel_error = (delta / real_val) * 100
                    rel_error_str = f"{rel_error:.2f}%"
                else:
                    rel_error_str = _("N/A")

                # Determine row style based on delta
                row_style = self._get_delta_style(delta)

                table.add_row(
                    resolution, real_str, ml_str, delta_str, rel_error_str,
                    style=row_style
                )

        # Add statistics row
        stats = comparison['statistics']
        median = stats['median_delta']
        mean = stats['mean_delta']

        if self.original_args.channels:
            table.add_row(
                _("Statistics"), "", "", "",
                f"{_('Median')}: {median:.3f}\n{_('Mean')}: {mean:.3f}", "",
                style="bold"
            )
        else:
            table.add_row(
                _("Statistics"), "", "",
                f"{_('Median')}: {median:.3f}\n{_('Mean')}: {mean:.3f}", "",
                style="bold"
            )

        console.print(table)
        console.print()

    def _print_overall_statistics(self) -> None:
        """Print overall statistics for all files."""
        console.print(Text(_("Overall Statistics"), style="bold cyan"))

        table = Table(show_header=True, header_style="bold")
        table.add_column(_("Statistic"), style="bold")
        table.add_column(_("Value"))

        max_file = self.overall_stats['max_median_delta_file']
        min_file = self.overall_stats['min_median_delta_file']

        table.add_row(_("Mean Median Delta"), f"{self.overall_stats['mean_median_delta']:.3f}")
        table.add_row(_("Max Median Delta"), f"{self.overall_stats['max_median_delta']:.3f}")
        table.add_row(_("File with Max Median Delta"), max_file if max_file else _("N/A"))
        table.add_row(_("Min Median Delta"), f"{self.overall_stats['min_median_delta']:.3f}")
        table.add_row(_("File with Min Median Delta"), min_file if min_file else _("N/A"))

        console.print(table)
        console.print()

    def _get_delta_style(self, delta: float) -> str:
        """Determine the style for a row based on the delta value."""
        if delta == float('inf'):
            return ""

        # Choose different thresholds based on the metric type
        metric_type = QualityMetrics(self.original_args.metric)

        if metric_type == QualityMetrics.PSNR:
            # For PSNR, higher differences are worse
            if delta < 1.0:
                return RICH_STYLES['excellent']
            elif delta < 3.0:
                return RICH_STYLES['very_good']
            elif delta < 5.0:
                return RICH_STYLES['good']
            else:
                return RICH_STYLES['poor']
        else:
            # For normalized metrics (SSIM, MS-SSIM, TDPR), scale differently
            if delta < 0.01:
                return RICH_STYLES['excellent']
            elif delta < 0.05:
                return RICH_STYLES['very_good']
            elif delta < 0.1:
                return RICH_STYLES['good']
            else:
                return RICH_STYLES['poor']