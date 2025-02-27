# utils/visualization.py
"""
Module for visualization of image quality analysis results.
"""
from ..i18n import _
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogFormatter
import matplotlib.colors as mcolors
from ..config import (
    QualityMetrics, LOGS_DIR, QUALITY_LEVEL_HINTS_DESCRIPTIONS,
    QualityLevelHints, QUALITY_METRIC_THRESHOLDS
)


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """
    Parses a resolution string in 'WxH' format and returns a tuple (width, height).

    Args:
        res_str: Resolution string in 'WxH' format

    Returns:
        Tuple (width, height)
    """
    match = re.match(r'(\d+)x(\d+)', res_str)
    if not match:
        raise ValueError(f"Invalid resolution format: {res_str}")
    return int(match.group(1)), int(match.group(2))


def get_megapixels(res_str: str) -> float:
    """
    Calculates the number of megapixels for a resolution.

    Args:
        res_str: Resolution string in 'WxH' format

    Returns:
        Number of megapixels
    """
    width, height = parse_resolution(res_str)
    return (width * height) / 1_000_000


def get_chart_filename(
    file_basename: str,
    metric_type: QualityMetrics,
    analyze_channels: bool = False
) -> str:
    """
    Creates a filename for the chart.

    Args:
        file_basename: Base name of the source file
        metric_type: Type of metric
        analyze_channels: Analysis by channels

    Returns:
        Path to the chart file
    """
    # Create directory for charts if it doesn't exist
    charts_dir = LOGS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Form the filename
    chart_filename = (
        f"{file_basename}_"
        f"{metric_type.value.upper()}"
        f"{'_channels' if analyze_channels else ''}"
        f".png"
    )

    return str(charts_dir / chart_filename)


def get_quality_color_map(metric_type: QualityMetrics) -> Tuple[Dict, List]:
    """
    Creates a colormap for quality levels.

    Args:
        metric_type: The quality metric being used

    Returns:
        Tuple of (color_dict, boundary_values)
    """
    thresholds = QUALITY_METRIC_THRESHOLDS.get(metric_type, {})

    # Define colors for each quality level
    color_dict = {
        QualityLevelHints.ORIGINAL: "#1a5fb4",      # Strong Blue
        QualityLevelHints.EXCELLENT: "#26a269",     # Green
        QualityLevelHints.VERY_GOOD: "#98c379",     # Light Green
        QualityLevelHints.GOOD: "#e5c07b",          # Yellow
        QualityLevelHints.NOTICEABLE_LOSS: "#e06c75" # Red
    }

    # Get threshold values for the colormap boundaries
    boundaries = []
    for level in [QualityLevelHints.NOTICEABLE_LOSS, QualityLevelHints.GOOD,
                  QualityLevelHints.VERY_GOOD, QualityLevelHints.EXCELLENT]:
        if level in thresholds:
            boundaries.append(thresholds[level])

    return color_dict, sorted(boundaries)


def generate_quality_chart(
    results: list,
    output_path: str,
    title: str = _("Quality vs Resolution Relationship"),
    metric_type: QualityMetrics = QualityMetrics.PSNR,
    analyze_channels: bool = False,
    channels: Optional[List[str]] = None
) -> str:
    """
    Creates a chart showing quality vs resolution relationship.

    Args:
        results: List of analysis results
        output_path: Path to save the chart
        title: Chart title
        metric_type: Type of quality metric
        analyze_channels: Whether to analyze by channels
        channels: List of image channels

    Returns:
        Path to the saved chart
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set up plotting style
    plt.style.use('ggplot')

    # Create the main figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # Get quality color mapping for this metric
    quality_colors, boundaries = get_quality_color_map(metric_type)

    # Skip original (with infinite quality)
    filtered_results = [r for r in results if QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL] not in r[-1]]

    if not filtered_results:
        # No results to display
        plt.close(fig)
        return output_path

    # Extract data
    resolutions = [r[0] for r in filtered_results]

    # Convert resolutions to pixel counts for X-axis
    pixel_counts = [get_megapixels(res) * 1_000_000 for res in resolutions]

    # Set up markers and colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    # Set up the main X-axis for pixel counts (logarithmic)
    ax.set_xscale('log')

    # Format the x-axis for readability
    def format_pixels(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{int(x)}'

    ax.xaxis.set_major_formatter(FuncFormatter(format_pixels))

    # Set initial y-axis limits
    min_y, max_y = float('inf'), float('-inf')

    # Add background shading for resolution sections
    # Calculate midpoints between pixel counts for section boundaries
    if len(pixel_counts) > 1:
        # Create logarithmic midpoints between pixel counts
        log_pixel_counts = np.log10(pixel_counts)
        section_boundaries = [10 ** (0.5 * (log_pixel_counts[i] + log_pixel_counts[i+1]))
                             for i in range(len(log_pixel_counts)-1)]

        # Add extreme boundaries
        section_boundaries = [pixel_counts[0] * 0.5] + section_boundaries + [pixel_counts[-1] * 2]

        # Create alternating background colors
        for i in range(len(section_boundaries) - 1):
            if i % 2 == 0:
                ax.axvspan(section_boundaries[i], section_boundaries[i+1],
                          alpha=0.1, color='gray', zorder=0)

    # Plot the data
    if analyze_channels and channels:
        # Plot for each channel
        channel_colors = {
            'R': '#e74c3c',  # Red
            'G': '#2ecc71',  # Green
            'B': '#3498db',  # Blue
            'A': '#9b59b6',  # Purple
            'L': '#34495e'   # Dark Gray (for luminance)
        }

        # Track which channel has the minimum value at each resolution
        min_channels = []
        for r in filtered_results:
            channel_dict = r[1]
            min_val = r[2]
            min_channel = None

            # Find which channel has the minimum value
            for ch in channels:
                if ch in channel_dict and abs(channel_dict[ch] - min_val) < 1e-6:
                    min_channel = ch
                    break

            min_channels.append(min_channel)

        # Plot each channel
        for i, channel in enumerate(channels):
            if channel not in channel_colors:
                continue

            values = []
            is_min_at_point = []

            for j, r in enumerate(filtered_results):
                channel_dict = r[1]
                value = channel_dict.get(channel, np.nan)
                if value == float('inf'):
                    value = np.nan
                values.append(value)

                # Check if this channel is the minimum at this point
                is_min_at_point.append(channel == min_channels[j])

                if not np.isnan(value):
                    min_y = min(min_y, value)
                    max_y = max(max_y, value)

            # Plot channel values
            line = ax.plot(
                pixel_counts,
                values,
                label=f"{channel}",
                marker=markers[i % len(markers)],
                color=channel_colors[channel],
                linewidth=2,
                markersize=8,
                alpha=0.8
            )

            # Highlight minimum values with larger markers and darker color
            for j, (x, y, is_min) in enumerate(zip(pixel_counts, values, is_min_at_point)):
                if is_min and not np.isnan(y):
                    ax.plot(
                        x, y,
                        marker=markers[i % len(markers)],
                        markersize=12,
                        markerfacecolor=channel_colors[channel],
                        markeredgecolor='black',
                        markeredgewidth=1.5,
                        alpha=1.0
                    )

        # Add quality level indicators to the points
        for j, (x, y, hint) in enumerate(zip(pixel_counts, [r[2] for r in filtered_results], [r[3] for r in filtered_results])):
            if np.isnan(y):
                continue

            # Find which quality level this belongs to
            hint_level = None
            for level, desc in QUALITY_LEVEL_HINTS_DESCRIPTIONS.items():
                if desc == hint:
                    hint_level = level
                    break

            if hint_level and hint_level in quality_colors:
                # Add a colored ring around the point to indicate quality level
                ax.plot(
                    x, y,
                    'o',
                    markersize=16,
                    markerfacecolor='none',
                    markeredgecolor=quality_colors[hint_level],
                    markeredgewidth=2,
                    alpha=0.7
                )
    else:
        # Single graph for overall metric
        metric_values = []
        hints = []

        for r in filtered_results:
            val = r[1]
            hint = r[2]
            hints.append(hint)

            if val == float('inf'):
                val = np.nan
            metric_values.append(val)

            if not np.isnan(val):
                min_y = min(min_y, val)
                max_y = max(max_y, val)

        # Draw lines between points
        line = ax.plot(
            pixel_counts,
            metric_values,
            label=f"{metric_type.value.upper()}",
            linewidth=2.5,
            color='#2980b9',
            alpha=0.8
        )

        # Add colorful points based on quality level
        for i, (x, y, hint) in enumerate(zip(pixel_counts, metric_values, hints)):
            if np.isnan(y):
                continue

            # Find which quality level this belongs to
            hint_level = None
            for level, desc in QUALITY_LEVEL_HINTS_DESCRIPTIONS.items():
                if desc == hint:
                    hint_level = level
                    break

            if hint_level and hint_level in quality_colors:
                ax.plot(
                    x, y,
                    'o',
                    markersize=12,
                    markerfacecolor=quality_colors[hint_level],
                    markeredgecolor='black',
                    alpha=0.8
                )

    # Set up y-axis limits
    if min_y == float('inf') or max_y == float('-inf'):
        if metric_type == QualityMetrics.PSNR:
            min_y, max_y = 0, 50
        else:
            min_y, max_y = 0, 1.0

    # Add padding to y-axis
    y_padding = (max_y - min_y) * 0.1
    y_min = max(0, min_y - y_padding)

    # For PSNR, make the upper limit adaptive to the data
    if metric_type == QualityMetrics.PSNR:
        # Add more headroom for PSNR values
        y_max = max_y + y_padding * 2

        # Ensure we show at least up to the Excellent threshold
        excellent_threshold = QUALITY_METRIC_THRESHOLDS[QualityMetrics.PSNR][QualityLevelHints.EXCELLENT]
        y_max = max(y_max, excellent_threshold * 1.2)
    else:
        # For normalized metrics (SSIM, MS-SSIM), always show the full quality range if possible
        # Check if we have any excellent quality points
        has_excellent = any(hint == QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.EXCELLENT]
                           for hint in [r[-1] for r in filtered_results])

        # If we have excellent quality points, show the full range up to 1.0
        if has_excellent:
            y_max = 1.05
        else:
            # Otherwise, just add padding to the max value
            y_max = max_y + y_padding
            y_max = min(y_max, 1.05)  # But don't exceed 1.05

    ax.set_ylim(y_min, y_max)

    # Add quality threshold lines
    thresholds = QUALITY_METRIC_THRESHOLDS.get(metric_type, {})

    # Get quality thresholds to display
    quality_levels = [
        (QualityLevelHints.EXCELLENT, _("Excellent")),
        (QualityLevelHints.VERY_GOOD, _("Very Good")),
        (QualityLevelHints.GOOD, _("Good")),
        (QualityLevelHints.NOTICEABLE_LOSS, _("noticeable_loss")),
    ]

    # Track which thresholds are visible in the chart
    visible_thresholds = []

    for level, label in quality_levels:
        if level in thresholds:
            threshold = thresholds[level]
            if y_min <= threshold <= y_max:
                ax.axhline(
                    y=threshold,
                    color=quality_colors[level],
                    linestyle='--',
                    alpha=0.7,
                    linewidth=1.5
                )
                visible_thresholds.append(level)

                # Add label to threshold line
                ax.text(
                    max(pixel_counts) * 1.15,
                    threshold,
                    f"{label}",
                    va='center',
                    ha='left',
                    fontsize=9,
                    color=quality_colors[level],
                    weight='bold'
                )

    # Add legend for channels if they're plotted
    if analyze_channels and channels:
        legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add quality color legend - but only for visible thresholds and used quality levels
    quality_patches = []

    # Get all quality levels used in the data
    used_quality_levels = set()
    for r in filtered_results:
        hint = r[-1]
        for level, desc in QUALITY_LEVEL_HINTS_DESCRIPTIONS.items():
            if desc == hint:
                used_quality_levels.add(level)
                break

    # Add patches for all visible thresholds and used quality levels
    for level, label in [
        (QualityLevelHints.EXCELLENT, _("Excellent")),
        (QualityLevelHints.VERY_GOOD, _("Very Good")),
        (QualityLevelHints.GOOD, _("Good")),
        (QualityLevelHints.NOTICEABLE_LOSS, _("Noticeable Loss"))
    ]:
        if level in quality_colors and (level in visible_thresholds or level in used_quality_levels):
            quality_patches.append(
                Patch(facecolor=quality_colors[level], edgecolor='black', label=label)
            )

    if quality_patches:
        quality_legend = ax.legend(
            handles=quality_patches,
            loc='lower left' if not analyze_channels else 'lower right',
            title=_("Quality Levels"),
            fontsize=10,
            framealpha=0.9
        )

        # If we have two legends, make sure they both show
        if analyze_channels and channels:
            ax.add_artist(legend)

    # Set axis labels
    ax.set_xlabel(_('Total Pixels (log scale)'), fontsize=12, fontweight='bold')

    if metric_type == QualityMetrics.PSNR:
        ylabel = f"{metric_type.value.upper()} (dB)"
    else:
        ylabel = f"{metric_type.value.upper()}"

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Set title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add secondary x-axis with resolutions
    secax = ax.twiny()
    secax.set_xscale('log')
    secax.set_xlim(ax.get_xlim())

    # Show all resolution points on the secondary axis
    secax.set_xticks(pixel_counts)
    secax.set_xticklabels(resolutions, rotation=45)
    secax.set_xlabel(_('Resolution'), fontsize=12, fontweight='bold', labelpad=10)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path