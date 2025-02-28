# utils/visualization.py
"""
Module for visualization of image quality analysis results.
"""
from ..i18n import _
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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
    charts_dir = LOGS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_filename = (
        f"{file_basename}_"
        f"{metric_type.value.upper()}"
        f"{'_ch' if analyze_channels else ''}"
        f".png"
    )

    return str(charts_dir / chart_filename)


def get_quality_color_map(metric_type: QualityMetrics) -> Tuple[Dict, List]:
    """
    Creates a colormap for quality levels.

    Args:
        metric_type: Type of quality metric

    Returns:
        Tuple containing a dictionary of colors by quality level and a list of boundaries
    """
    thresholds = QUALITY_METRIC_THRESHOLDS.get(metric_type, {})

    # Define colors for each quality level
    color_dict = {
        QualityLevelHints.ORIGINAL: "#1a5fb4",
        QualityLevelHints.EXCELLENT: "#26a269",
        QualityLevelHints.VERY_GOOD: "#98c379",
        QualityLevelHints.GOOD: "#e5c07b",
        QualityLevelHints.NOTICEABLE_LOSS: "#e06c75"
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup figure and axes
    fig, ax = _setup_figure_and_axes()

    # Process data
    filtered_results = _filter_results(results)
    if not filtered_results:
        plt.close(fig)
        return output_path

    pixel_counts, resolutions, metric_values, hints = _prepare_data(filtered_results)

    # Get quality color mapping for this metric
    quality_colors, boundaries = get_quality_color_map(metric_type)

    # Setup background shading
    _setup_background_shading(ax, pixel_counts)

    # Plot data
    min_y, max_y = _plot_data(
        ax, pixel_counts, filtered_results, metric_type,
        analyze_channels, channels, quality_colors
    )

    # Configure axes
    _configure_y_axis(ax, min_y, max_y, metric_type, filtered_results)

    # Add quality threshold lines
    visible_thresholds = _add_quality_thresholds(ax, metric_type, quality_colors)

    # Add legends
    _add_legends(ax, analyze_channels, channels, quality_colors,
                filtered_results, metric_type, visible_thresholds)

    # Configure secondary axis (resolution)
    sec_ax = _add_secondary_axis(ax, pixel_counts, resolutions)
    # Critical fix: prevent secondary axis grid from overlapping
    sec_ax.grid(False)

    # Set labels and title
    _set_labels_and_title(ax, title, metric_type)

    # Save and cleanup
    return _save_chart(fig, output_path)


def _setup_figure_and_axes() -> Tuple[Figure, Axes]:
    """Create and setup the figure and axes."""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffff')
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)  # Add grid with low z-order
    return fig, ax


def _filter_results(results: list) -> list:
    """Filter out original results with infinite quality."""
    return [r for r in results if QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL] not in r[-1]]


def _prepare_data(filtered_results: list) -> Tuple[List[float], List[str], List[float], List[str]]:
    """Extract and prepare data from filtered results."""
    resolutions = [r[0] for r in filtered_results]
    pixel_counts = [get_megapixels(res) * 1_000_000 for res in resolutions]

    # For non-channel analysis
    metric_values = []
    hints = []

    for r in filtered_results:
        val = r[1] if isinstance(r[1], (int, float)) else r[2]
        hint = r[2] if isinstance(r[1], dict) else r[-1]
        hints.append(hint)

        if val == float('inf'):
            val = np.nan
        metric_values.append(val)

    return pixel_counts, resolutions, metric_values, hints


def _setup_background_shading(ax: Axes, pixel_counts: List[float]) -> None:
    """Add background shading for resolution sections."""
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


def _format_pixels(x: float, pos) -> str:
    """Format pixel counts for readability."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{int(x)}'


def _identify_min_channels(filtered_results: list, channels: List[str]) -> List[Optional[str]]:
    """Identify which channel has the minimum value at each resolution."""
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

    return min_channels


def _extract_channel_values(
    filtered_results: list,
    channel: str,
    min_channels: List[Optional[str]]
) -> Tuple[List[float], List[bool]]:
    """Extract values for a specific channel and identify minimum points."""
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

    return values, is_min_at_point


def _highlight_min_values(
    ax: Axes,
    pixel_counts: List[float],
    values: List[float],
    is_min_at_point: List[bool],
    marker: str,
    color: str
) -> None:
    """Highlight minimum values with larger markers and darker color."""
    for j, (x, y, is_min) in enumerate(zip(pixel_counts, values, is_min_at_point)):
        if is_min and not np.isnan(y):
            ax.plot(
                x, y,
                marker=marker,
                markersize=12,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=1.5,
                alpha=1.0
            )


def _add_quality_indicators(
    ax: Axes,
    pixel_counts: List[float],
    filtered_results: list,
    quality_colors: Dict
) -> None:
    """Add quality level indicators to data points."""
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


def _plot_channel_data(
    ax: Axes,
    pixel_counts: List[float],
    filtered_results: list,
    channels: List[str],
    quality_colors: Dict
) -> Tuple[float, float]:
    """Plot data for individual channels."""
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    min_y, max_y = float('inf'), float('-inf')

    # Define channel colors
    channel_colors = {
        'R': '#e74c3c',  # Red
        'G': '#2ecc71',  # Green
        'B': '#3498db',  # Blue
        'A': '#9b59b6',  # Purple
        'L': '#34495e'   # Dark Gray (for luminance)
    }

    # Identify minimum value channels at each resolution
    min_channels = _identify_min_channels(filtered_results, channels)

    # Plot each channel
    for i, channel in enumerate(channels):
        if channel not in channel_colors:
            continue

        values, is_min_at_point = _extract_channel_values(
            filtered_results, channel, min_channels
        )

        # Update min/max values
        for val in values:
            if not np.isnan(val):
                min_y = min(min_y, val)
                max_y = max(max_y, val)

        # Plot channel line
        ax.plot(
            pixel_counts,
            values,
            label=f"{channel}",
            marker=markers[i % len(markers)],
            color=channel_colors[channel],
            linewidth=2,
            markersize=8,
            alpha=0.8
        )

        # Highlight minimum values
        _highlight_min_values(ax, pixel_counts, values, is_min_at_point,
                             markers[i % len(markers)], channel_colors[channel])

    # Add quality indicators
    _add_quality_indicators(ax, pixel_counts, filtered_results, quality_colors)

    return min_y, max_y


def _plot_metric_data(
    ax: Axes,
    pixel_counts: List[float],
    metric_values: List[float],
    hints: List[str],
    metric_type: QualityMetrics,
    quality_colors: Dict
) -> Tuple[float, float]:
    """Plot single metric data."""
    min_y, max_y = float('inf'), float('-inf')

    # Update min/max values
    for val in metric_values:
        if not np.isnan(val):
            min_y = min(min_y, val)
            max_y = max(max_y, val)

    # Draw lines between points
    ax.plot(
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

    return min_y, max_y


def _plot_data(
    ax: Axes,
    pixel_counts: List[float],
    filtered_results: list,
    metric_type: QualityMetrics,
    analyze_channels: bool,
    channels: Optional[List[str]],
    quality_colors: Dict
) -> Tuple[float, float]:
    """Plot data points on the chart."""
    # Setup axes
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(_format_pixels))

    if analyze_channels and channels:
        return _plot_channel_data(
            ax, pixel_counts, filtered_results, channels, quality_colors
        )
    else:
        metric_values = []
        hints = []

        for r in filtered_results:
            val = r[1]
            hint = r[2]
            hints.append(hint)

            if val == float('inf'):
                val = np.nan
            metric_values.append(val)

        return _plot_metric_data(ax, pixel_counts, metric_values, hints, metric_type, quality_colors)


def _configure_y_axis(ax: Axes, min_y: float, max_y: float, metric_type: QualityMetrics, filtered_results: list) -> None:
    """Configure the y-axis based on metric type and data range."""
    # Set up y-axis limits
    if min_y == float('inf') or max_y == float('-inf'):
        if metric_type == QualityMetrics.PSNR:
            min_y, max_y = 0, 50
        else:
            min_y, max_y = 0, 1.0

    # Add padding to y-axis
    y_padding = (max_y - min_y) * 0.1
    y_min = max(0.0, min_y - y_padding)

    # Get the threshold for noticeable loss to ensure it's visible
    noticeable_loss_threshold = QUALITY_METRIC_THRESHOLDS[metric_type][QualityLevelHints.NOTICEABLE_LOSS]

    # For PSNR, make the upper limit adaptive to the data
    if metric_type == QualityMetrics.PSNR:
        # Add more headroom for PSNR values
        y_max = max_y + y_padding * 2

        # Ensure we show at least up to the Noticeable Loss threshold
        y_max = max(y_max, noticeable_loss_threshold * 1.2)
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

        # Ensure the noticeable loss threshold is visible
        y_max = max(y_max, noticeable_loss_threshold * 1.2)

    # Ensure y_min is below the data points
    y_min = min(y_min, min_y * 0.9)

    # Set the y-axis limits
    ax.set_ylim(y_min, y_max)


def _add_quality_thresholds(ax: Axes, metric_type: QualityMetrics, quality_colors: Dict) -> List:
    """Add quality threshold lines to the chart."""
    thresholds = QUALITY_METRIC_THRESHOLDS.get(metric_type, {})

    # Get quality thresholds to display
    quality_levels = [
        (QualityLevelHints.EXCELLENT, _("excellent")),
        (QualityLevelHints.VERY_GOOD, _("very_good")),
        (QualityLevelHints.GOOD, _("good")),
        (QualityLevelHints.NOTICEABLE_LOSS, _("noticeable_loss")),
    ]

    # Track which thresholds are visible in the chart
    visible_thresholds = []

    y_min, y_max = ax.get_ylim()

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

    return visible_thresholds


def _add_legends(
    ax: Axes,
    analyze_channels: bool,
    channels: Optional[List[str]],
    quality_colors: Dict,
    filtered_results: list,
    metric_type: QualityMetrics,
    visible_thresholds: List
) -> None:
    """Add appropriate legends to the chart."""
    # Add legend for channels if they're plotted
    if analyze_channels and channels:
        legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Get all quality levels used in the data
    used_quality_levels = set()
    for r in filtered_results:
        hint = r[-1]
        for level, desc in QUALITY_LEVEL_HINTS_DESCRIPTIONS.items():
            if desc == hint:
                used_quality_levels.add(level)
                break

    # Get quality levels for legend
    quality_levels = [
        (QualityLevelHints.EXCELLENT, QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.EXCELLENT]),
        (QualityLevelHints.VERY_GOOD, QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.VERY_GOOD]),
        (QualityLevelHints.GOOD, QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.GOOD]),
        (QualityLevelHints.NOTICEABLE_LOSS, QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.NOTICEABLE_LOSS]),
    ]

    # Add patches for visible thresholds and used quality levels
    quality_patches = []
    for level, label in quality_levels:
        if level in quality_colors and (level in visible_thresholds or level in used_quality_levels):
            quality_patches.append(
                Patch(facecolor=quality_colors[level], edgecolor='black', label=label)
            )

    if quality_patches:
        quality_legend = ax.legend(
            handles=quality_patches,
            loc='lower right',
            title=_("Quality Levels"),
            fontsize=10,
            framealpha=0.9
        )

        # If we have two legends, make sure they both show
        if analyze_channels and channels:
            ax.add_artist(legend)


def _add_secondary_axis(ax: Axes, pixel_counts: List[float], resolutions: List[str]) -> Axes:
    """Add secondary x-axis with resolutions."""
    sec_ax = ax.twiny()
    sec_ax.set_xscale('log')
    sec_ax.set_xlim(ax.get_xlim())

    # Show all resolution points on the secondary axis
    sec_ax.set_xticks(pixel_counts)
    sec_ax.set_xticklabels(resolutions, rotation=45)
    sec_ax.set_xlabel(_('Resolution'), fontsize=12, fontweight='bold', labelpad=10)
    sec_ax.grid(False)  # Explicitly disable grid on secondary axis

    return sec_ax


def _set_labels_and_title(ax: Axes, title: str, metric_type: QualityMetrics) -> None:
    """Set axis labels and chart title."""
    ax.set_xlabel(_('Total Pixels (log scale)'), fontsize=12, fontweight='bold')

    if metric_type == QualityMetrics.PSNR:
        y_label = f"{metric_type.value.upper()} (dB)"
    else:
        y_label = f"{metric_type.value.upper()}"

    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


def _save_chart(fig: Figure, output_path: str) -> str:
    """Save the chart to file and clean up."""
    plt.tight_layout()
    plt.savefig(output_path, dpi=96, bbox_inches='tight')
    plt.close(fig)
    return output_path