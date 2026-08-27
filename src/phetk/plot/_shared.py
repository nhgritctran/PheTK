"""Shared stateless helper functions used by multiple plot backends."""
import datetime
import adjustText
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

MARKER_MIN_SIZE = 13
MARKER_MAX_SIZE = 267


def save_plot(
        plot_type: str = "manhattan",
        output_file_path: str | None = None
) -> None:
    """
    Save current matplotlib plot to file with automatic filename generation.

    Creates timestamped filename if none provided and saves plot with
    specified file format and tight bounding box. File format is automatically
    detected from file extension or defaults to PDF.

    Args:
        plot_type: Type of plot for filename generation when auto-generating.
        output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"),
            auto-generated with timestamp if None.
    """
    if output_file_path is not None:
        # If no extension provided, default to PDF
        if "." not in output_file_path:
            output_file_path = output_file_path + ".pdf"
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"{plot_type}_{timestamp}.pdf"

    plt.savefig(output_file_path, bbox_inches="tight", pad_inches=0.3)
    print()
    print("Plot saved to", output_file_path)
    print()


def filter_by_phecode_categories(
        df: pl.DataFrame,
        phecode_categories: list[str] | str | None = None
) -> pl.DataFrame:
    """
    Filter PheWAS results by specified phecode categories.

    Args:
        df: PheWAS result dataframe to filter.
        phecode_categories: Specific phecode categories to include, uses all if None.

    Returns:
        Filtered dataframe containing only specified categories.
    """
    if phecode_categories:
        if isinstance(phecode_categories, str):
            phecode_categories = [phecode_categories]
        df = df.filter(pl.col("phecode_category").is_in(phecode_categories))
    return df


def compute_effect_sizes(
        df: pl.DataFrame,
        direction_col: str,
        min_size: float = MARKER_MIN_SIZE,
        max_size: float = MARKER_MAX_SIZE,
        effect_cap: float = 10,
) -> pl.DataFrame:
    """
    Compute effect magnitude and scaled marker sizes for plot backends.

    Uses odds ratio or hazard ratio when available, otherwise falls back
    to exp(abs(direction_col)). Values are capped at effect_cap and
    min-max normalized to [min_size, max_size].

    Args:
        df: PheWAS result dataframe.
        direction_col: Column name for effect direction.
        min_size: Minimum marker size.
        max_size: Maximum marker size.
        effect_cap: Cap for effect magnitude. Values above get max marker size.

    Returns:
        Dataframe with _effect_magnitude, _capped, and _scaled_size columns added.
    """
    if "odds_ratio" in df.columns:
        ratio_col = "odds_ratio"
    elif "hazard_ratio" in df.columns:
        ratio_col = "hazard_ratio"
    else:
        ratio_col = None

    if ratio_col is not None:
        df = df.with_columns(
            pl.max_horizontal(pl.col(ratio_col), 1.0 / pl.col(ratio_col)).alias("_effect_magnitude")
        )
    else:
        df = df.with_columns(
            pl.col(direction_col).abs().exp().alias("_effect_magnitude")
        )

    df = df.with_columns(
        (pl.col("_effect_magnitude") > effect_cap).alias("_capped"),
        pl.col("_effect_magnitude").clip(upper_bound=effect_cap).alias("_effect_magnitude"),
    )

    df = transform_values(df, "_effect_magnitude", "_scaled_size", new_min=min_size, new_max=max_size)

    return df


def split_by_effect_size(
        df: pl.DataFrame,
        direction_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split PheWAS results into positive and negative effect directions.

    Args:
        df: PheWAS result dataframe to split.
        direction_col: Column name indicating effect direction (e.g., "beta" or "log_hazard_ratio").

    Returns:
        Positive effects dataframe and negative effects dataframe.
    """
    positive_betas = df.filter(pl.col(direction_col) >= 0).sort(by=direction_col, descending=True)
    negative_betas = df.filter(pl.col(direction_col) < 0).sort(by=direction_col, descending=False)
    return positive_betas, negative_betas


def draw_lines(
        ax,
        plot_type: str,
        plot_df: pl.DataFrame,
        x_col: str,
        bonferroni: float,
        inf_proxy: float | None,
        offset: float,
        nominal_significance_line: bool = False,
        bonferroni_line: bool = False,
        infinity_line: bool = False,
        y_threshold_line: bool = False,
        y_threshold_value: float | None = None,
        x_positive_threshold_line: bool = False,
        x_positive_threshold_value: float | None = None,
        x_negative_threshold_line: bool = False,
        x_negative_threshold_value: float | None = None
) -> None:
    """
    Draw significance and threshold lines on plot.

    Args:
        ax: Matplotlib axes object for plotting.
        plot_type: Type of plot ("manhattan" or "volcano") for offset calculation.
        plot_df: Plot dataframe for determining line extent.
        x_col: Column name for x-axis values.
        bonferroni: Bonferroni correction threshold value.
        inf_proxy: Proxy value used for infinite p-values, or None.
        offset: Base offset for line extent.
        nominal_significance_line: Whether to draw nominal significance line (p=0.05).
        bonferroni_line: Whether to draw Bonferroni correction line.
        infinity_line: Whether to draw infinity proxy line.
        y_threshold_line: Whether to draw custom y-threshold line.
        y_threshold_value: Y-value for custom threshold line.
        x_positive_threshold_line: Whether to draw positive x-threshold line.
        x_positive_threshold_value: X-value for positive threshold line.
        x_negative_threshold_line: Whether to draw negative x-threshold line.
        x_negative_threshold_value: X-value for negative threshold line.
    """
    extra_offset = 0
    if plot_type == "manhattan":
        extra_offset = 1
    elif plot_type == "volcano":
        extra_offset = 0.05

    if nominal_significance_line:
        ax.hlines(
            y=-np.log10(.05),
            xmin=plot_df[x_col].min() - offset - extra_offset,
            xmax=plot_df[x_col].max() + offset + extra_offset,
            colors="#E8A0A0",
            linestyles="dotted",
            lw=0.8
        )

    if bonferroni_line:
        ax.hlines(
            y=bonferroni,
            xmin=plot_df[x_col].min() - offset - extra_offset,
            xmax=plot_df[x_col].max() + offset + extra_offset,
            colors="#82C882",
            linestyles="dashed",
            lw=0.8
        )

    if infinity_line:
        if inf_proxy is not None:
            ax.yaxis.get_major_ticks()[-2].set_visible(False)
            ax.hlines(
                y=inf_proxy * 0.98,
                xmin=plot_df[x_col].min() - offset - extra_offset,
                xmax=plot_df[x_col].max() + offset + extra_offset,
                colors="#A0A0E8",
                linestyle="dashdot",
                lw=0.8
            )

    if y_threshold_line:
        ax.hlines(
            y=y_threshold_value,
            xmin=plot_df[x_col].min() - offset - extra_offset,
            xmax=plot_df[x_col].max() + offset + extra_offset,
            colors="gray",
            linestyles="dashed",
            lw=1
        )

    if x_positive_threshold_line:
        ax.vlines(
            x=x_positive_threshold_value,
            ymin=plot_df["neg_log_p_value"].min() - offset,
            ymax=plot_df["neg_log_p_value"].max() + offset + 5,
            colors="orange",
            linestyles="dashed",
            lw=1
        )
    if x_negative_threshold_line:
        ax.vlines(
            x=x_negative_threshold_value,
            ymin=plot_df["neg_log_p_value"].min() - offset,
            ymax=plot_df["neg_log_p_value"].max() + offset + 5,
            colors="lightseagreen",
            linestyles="dashed",
            lw=1
        )


def split_text(s: str, threshold: int = 30) -> str:
    """
    Split long text labels into multiple lines for better readability.

    Args:
        s: Text string to split.
        threshold: Approximate number of characters per line.

    Returns:
        Text with line breaks inserted at appropriate positions.
    """
    words = s.split(" ")
    new_s = ""
    line_length = 0
    for word in words:
        new_s += word
        line_length += len(word)
        if line_length >= threshold and word != words[-1]:
            new_s += "\n"
            line_length = 0
        else:
            new_s += " "
    return new_s


def x_ticks(
        plot_df: pl.DataFrame,
        selected_color_dict: dict[str, str],
        size: int = 8
) -> None:
    """
    Generate colored x-axis tick labels for Manhattan plot.

    Args:
        plot_df: Plot dataframe containing phecode categories and indices.
        selected_color_dict: Color mapping for phecode categories.
        size: Font size for tick labels.
    """
    x_tick_data = plot_df[["phecode_category", "phecode_index"]].group_by("phecode_category").mean()
    adjustText.plt.xticks(
        x_tick_data["phecode_index"],
        x_tick_data["phecode_category"],
        rotation=45,
        ha="right",
        weight="normal",
        size=size
    )
    tick_labels = adjustText.plt.gca().get_xticklabels()
    sorted_labels = sorted(tick_labels, key=lambda label: label.get_text())
    for tick_label, tick_color in zip(sorted_labels, selected_color_dict.values()):
        tick_label.set_color(tick_color)


def transform_values(
        df: pl.DataFrame,
        col: str,
        new_col: str,
        new_min: float,
        new_max: float
) -> pl.DataFrame:
    """
    Transform column values to specified range using min-max normalization.

    Args:
        df: Input dataframe containing column to transform.
        col: Name of column to transform.
        new_col: Name for new column containing transformed values.
        new_min: Minimum value for transformed range.
        new_max: Maximum value for transformed range.

    Returns:
        Dataframe with additional column containing transformed values.
    """
    df = df.with_columns(
        (
            (pl.col(col) - pl.col(col).min())
            * (new_max - new_min)
            / (pl.col(col).max() - pl.col(col).min())
            + new_min
        ).alias(new_col))
    return df
