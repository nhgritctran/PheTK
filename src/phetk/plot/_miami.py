"""Miami plot backend — mirrored Manhattan plot with effect-magnitude-sized circles."""
import adjustText
import matplotlib.colors as mc
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from phetk.plot._base import PlotBackend, PlotContext, register
from phetk.plot._manhattan import _create_phecode_index
from phetk.plot._shared import (
    MARKER_MAX_SIZE,
    MARKER_MIN_SIZE,
    compute_effect_sizes,
    filter_by_phecode_categories,
    save_plot,
    split_by_effect_size,
    split_text,
    x_ticks,
)


@register("miami")
class MiamiBackend(PlotBackend):
    """Backend for Miami (mirrored Manhattan) plot visualization."""

    def render(self, ctx: PlotContext, **kwargs) -> None:
        """
        Create Miami plot visualization of PheWAS results.

        Args:
            ctx: Shared plot context with data and settings.
            **kwargs: All Miami plot parameters (see Plot.miami signature).
        """
        # Extract kwargs
        label_values = kwargs.get("label_values", "p_value")
        label_value_threshold = kwargs.get("label_value_threshold", 0)
        label_count = kwargs.get("label_count", 10)
        label_size = kwargs.get("label_size", 8)
        label_text_column = kwargs.get("label_text_column", "phecode_string")
        label_color = kwargs.get("label_color", "label_color")
        label_weight = kwargs.get("label_weight", "normal")
        label_split_threshold = kwargs.get("label_split_threshold", 30)
        label_box_alpha = kwargs.get("label_box_alpha", 0.5)
        marker_alpha = kwargs.get("marker_alpha", 0.7)
        positive_marker_alpha = kwargs.get("positive_marker_alpha", None)
        negative_marker_alpha = kwargs.get("negative_marker_alpha", None)
        hide_non_significant = kwargs.get("hide_non_significant", False)
        marker_min_size = kwargs.get("marker_min_size", MARKER_MIN_SIZE)
        marker_max_size = kwargs.get("marker_max_size", MARKER_MAX_SIZE)
        phecode_categories = kwargs.get("phecode_categories", None)
        plot_all_categories = kwargs.get("plot_all_categories", True)
        sort_by_significance = kwargs.get("sort_by_significance", False)
        title = kwargs.get("title", None)
        title_text_size = kwargs.get("title_text_size", 10)
        y_limit = kwargs.get("y_limit", None)
        axis_text_size = kwargs.get("axis_text_size", 8)
        show_legend = kwargs.get("show_legend", True)
        show_size_legend = kwargs.get("show_size_legend", True)
        size_legend_count = kwargs.get("size_legend_count", 4)
        legend_marker_size = kwargs.get("legend_marker_size", 7)
        dpi = kwargs.get("dpi", 150)
        do_save_plot = kwargs.get("save_plot", True)
        output_file_path = kwargs.get("output_file_path", None)

        # Local state
        offset = 9

        # phecode_categories & label_categories
        if phecode_categories:
            if isinstance(phecode_categories, str):
                phecode_categories = [phecode_categories]
            phecode_categories.sort()
            label_categories = phecode_categories
        else:
            label_categories = None

        # plot_df and color dict
        if plot_all_categories:
            selected_color_dict = ctx.color_dict
            n_categories = len(ctx.phewas_result.columns)
            plot_df = _create_phecode_index(ctx.phewas_result, sort_by_significance)
        else:
            if phecode_categories:
                selected_color_dict = {k: ctx.color_dict[k] for k in phecode_categories}
                n_categories = len(phecode_categories)
                dpi = None
                plot_df = _create_phecode_index(
                    filter_by_phecode_categories(ctx.phewas_result, phecode_categories=phecode_categories),
                    sort_by_significance,
                )
            else:
                print("phecode_categories must not be None when plot_all_categories = False.")
                return

        # Compute effect magnitude and marker sizes
        effect_cap = kwargs.get("effect_cap", 10)
        capped_marker_style = kwargs.get("capped_marker_style", "circle")
        plot_df = compute_effect_sizes(
            plot_df, ctx.direction_col,
            min_size=marker_min_size, max_size=marker_max_size,
            effect_cap=effect_cap,
        )
        has_capped = plot_df["_capped"].any()

        # Compute mirrored y-values: neg_log_p_value * sign(direction_col)
        plot_df = plot_df.with_columns(
            (pl.col("neg_log_p_value") * pl.col(ctx.direction_col).sign()).alias("_miami_y")
        )

        # Per-point alpha: use per-half overrides if provided, else unified marker_alpha
        pos_alpha = positive_marker_alpha if positive_marker_alpha is not None else marker_alpha
        neg_alpha = negative_marker_alpha if negative_marker_alpha is not None else marker_alpha
        plot_df = plot_df.with_columns(
            pl.when(pl.col(ctx.direction_col) >= 0)
            .then(pl.lit(pos_alpha))
            .otherwise(pl.lit(neg_alpha))
            .alias("_alpha")
        )

        # Optionally drop points below nominal significance
        if hide_non_significant:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") >= ctx.nominal_significance)

        # Create plot
        ratio = n_categories / len(ctx.phewas_result.columns)
        fig, ax = adjustText.plt.subplots(figsize=(12 * ratio, 7), dpi=dpi)

        # Plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # Filter data by y_limit if specified (applies to both halves symmetrically)
        if y_limit is not None:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") <= y_limit)
            ax.set_ylim(-y_limit, y_limit)

        # y axis label
        ax.set_ylabel(r"$-\log_{10}$(p-value) $\times$ sign(effect)", size=axis_text_size)

        # Split into positive & negative betas
        positive_betas, negative_betas = split_by_effect_size(plot_df, ctx.direction_col)

        ############
        # PLOTTING #
        ############

        # x-axis offset
        adjustText.plt.xlim(float(plot_df["phecode_index"].min()) - offset - 1,
                            float(plot_df["phecode_index"].max()) + offset + 1)

        # x ticks
        x_ticks(plot_df, selected_color_dict)

        # scatter
        _miami_scatter(ax, positive_betas, negative_betas, capped_marker_style=capped_marker_style)

        # lines
        _miami_lines(ax, plot_df, ctx.bonferroni, ctx.nominal_significance, ctx.inf_proxy, offset)

        # labeling
        _miami_label(
            plot_df=plot_df,
            positive_betas=positive_betas,
            negative_betas=negative_betas,
            direction_col=ctx.direction_col,
            label_values=label_values,
            label_count=label_count,
            label_text_column=label_text_column,
            label_categories=label_categories,
            label_value_threshold=label_value_threshold,
            label_split_threshold=label_split_threshold,
            label_size=label_size,
            label_color=label_color,
            label_weight=label_weight,
            label_box_alpha=label_box_alpha,
        )

        # legend
        if show_legend:
            _miami_legend(
                ax, plot_df, legend_marker_size, ctx.inf_proxy,
                show_size_legend, size_legend_count,
                marker_min_size, marker_max_size,
                effect_cap, has_capped, capped_marker_style,
            )

        # save plot
        if do_save_plot:
            save_plot(plot_type="miami", output_file_path=output_file_path)


def _miami_scatter(
        ax,
        positive_betas: pl.DataFrame,
        negative_betas: pl.DataFrame,
        capped_marker_style: str = "circle",
) -> None:
    """
    Generate scatter plot points for Miami plot with circle markers sized by effect magnitude.

    Each point's alpha is set individually from the ``_alpha`` column.
    When ``capped_marker_style="diamond"``, capped points are drawn as diamonds;
    otherwise all points are uniform circles.

    Args:
        ax: Matplotlib axes object for plotting.
        positive_betas: Dataframe of positive effect results.
        negative_betas: Dataframe of negative effect results.
        capped_marker_style: "circle" for uniform circles, "diamond" for diamond-shaped capped markers.
    """
    for half in (positive_betas, negative_betas):
        if capped_marker_style == "diamond":
            groups = [
                (half.filter(~pl.col("_capped")), "o"),
                (half.filter(pl.col("_capped")), "D"),
            ]
        else:
            groups = [(half, "o")]

        for subset, marker in groups:
            if len(subset) == 0:
                continue
            # Build per-point RGBA with significance-based alpha
            rgba = np.array([
                mc.to_rgba(c, alpha=a)
                for c, a in zip(subset["label_color"].to_list(), subset["_alpha"].to_list())
            ])
            ax.scatter(
                x=subset["phecode_index"].to_numpy(),
                y=subset["_miami_y"].to_numpy(),
                s=subset["_scaled_size"].to_numpy(),
                c=rgba,
                marker=marker,
            )


def _miami_lines(
        ax,
        plot_df: pl.DataFrame,
        bonferroni: float,
        nominal_significance: float,
        inf_proxy: float | None,
        offset: float,
) -> None:
    """
    Draw mirrored significance and reference lines on Miami plot.

    Args:
        ax: Matplotlib axes object for plotting.
        plot_df: Plot dataframe for determining line extent.
        bonferroni: Bonferroni correction threshold value.
        nominal_significance: Nominal significance value (-log10(0.05)).
        inf_proxy: Proxy value used for infinite p-values, or None.
        offset: Offset for line extent beyond data range.
    """
    xmin = plot_df["phecode_index"].min() - offset - 1
    xmax = plot_df["phecode_index"].max() + offset + 1

    # y = 0 reference line (divider)
    ax.hlines(y=0, xmin=xmin, xmax=xmax, colors="black", lw=0.5, linestyles="solid")

    # Bonferroni on both sides
    ax.hlines(y=bonferroni, xmin=xmin, xmax=xmax, colors="#82C882", linestyles="dashed", lw=0.8)
    ax.hlines(y=-bonferroni, xmin=xmin, xmax=xmax, colors="#82C882", linestyles="dashed", lw=0.8)

    # Nominal significance on both sides
    ax.hlines(y=nominal_significance, xmin=xmin, xmax=xmax, colors="#E8A0A0", linestyles="dotted", lw=0.8)
    ax.hlines(y=-nominal_significance, xmin=xmin, xmax=xmax, colors="#E8A0A0", linestyles="dotted", lw=0.8)

    # Infinity proxy on both sides
    if inf_proxy is not None:
        proxy_line = inf_proxy * 0.98
        ax.hlines(y=proxy_line, xmin=xmin, xmax=xmax, colors="#A0A0E8", linestyle="dashdot", lw=0.8)
        ax.hlines(y=-proxy_line, xmin=xmin, xmax=xmax, colors="#A0A0E8", linestyle="dashdot", lw=0.8)


def _miami_label(
        plot_df: pl.DataFrame,
        positive_betas: pl.DataFrame,
        negative_betas: pl.DataFrame,
        direction_col: str,
        label_values: str | list[str],
        label_count: int,
        label_categories: list[str] | None = None,
        label_text_column: str = "phecode_string",
        label_value_threshold: float = 0,
        label_split_threshold: int = 30,
        label_color: str = "label_color",
        label_size: int = 8,
        label_weight: str = "normal",
        label_box_alpha: float = 0.5,
):
    """
    Add data point labels to Miami plot with automatic positioning.

    Uses mirrored y-values (_miami_y) for label placement so labels
    appear on the correct half of the plot.

    Args:
        plot_df: Plot dataframe containing results to label.
        positive_betas: Positive effect dataframe.
        negative_betas: Negative effect dataframe.
        direction_col: Column name for effect direction.
        label_values: Labeling criteria.
        label_count: Maximum number of items to label.
        label_categories: Specific phecode categories to restrict labeling to.
        label_text_column: Column containing text for labels.
        label_value_threshold: Threshold value for filtering labels.
        label_split_threshold: Character count threshold for splitting long labels.
        label_color: Color specification or column name containing colors.
        label_size: Font size for labels.
        label_weight: Font weight for labels.
        label_box_alpha: Alpha value for label background boxes.
    """
    if isinstance(label_values, str):
        label_values = [label_values]

    data_to_label = pl.DataFrame(schema=plot_df.schema)

    for item in label_values:
        if item == "positive_beta":
            subset = positive_betas.filter(pl.col(direction_col) >= label_value_threshold)
            if label_categories is not None:
                subset = subset.filter(pl.col("phecode_category").is_in(label_categories))
            data_to_label = pl.concat([data_to_label, subset[:label_count]])
        elif item == "negative_beta":
            subset = negative_betas.filter(pl.col(direction_col) <= label_value_threshold)
            if label_categories is not None:
                subset = subset.filter(pl.col("phecode_category").is_in(label_categories))
            data_to_label = pl.concat([data_to_label, subset[:label_count]])
        elif item == "p_value":
            subset = plot_df.sort(by="p_value").filter(
                pl.col("neg_log_p_value") >= label_value_threshold
            )
            if label_categories is not None:
                subset = subset.filter(pl.col("phecode_category").is_in(label_categories))
            data_to_label = pl.concat([data_to_label, subset[:label_count]])
        else:
            data_to_label = pl.concat([data_to_label, plot_df.filter(pl.col("phecode") == item)])

    texts = []
    for i in range(len(data_to_label)):
        if mc.is_color_like(label_color):
            color = pl.Series(values=[label_color] * len(data_to_label))
        else:
            color = data_to_label[label_color]
        texts.append(
            adjustText.plt.text(
                float(data_to_label["phecode_index"][i]),
                float(data_to_label["_miami_y"][i]),
                split_text(data_to_label[label_text_column][i], label_split_threshold),
                color=color[i],
                size=label_size,
                weight=label_weight,
                alpha=1,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    boxstyle="round",
                    alpha=label_box_alpha,
                    lw=0.5,
                ),
            )
        )

    if len(texts) > 0:
        return adjustText.adjust_text(
            texts,
            arrowprops=dict(arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2),
        )


def _miami_legend(
        ax,
        plot_df: pl.DataFrame,
        legend_marker_size: int,
        inf_proxy: float | None,
        show_size_legend: bool,
        size_legend_count: int,
        marker_min_size: float,
        marker_max_size: float,
        effect_cap: float = 10,
        has_capped: bool = False,
        capped_marker_style: str = "circle",
) -> None:
    """
    Create legend for Miami plot with line descriptions and optional size reference.

    Args:
        ax: Matplotlib axes object for legend.
        plot_df: Plot dataframe for computing size legend values.
        legend_marker_size: Font size for legend text.
        inf_proxy: Infinity proxy value, or None if no infinity values.
        show_size_legend: Whether to include size reference circles.
        size_legend_count: Number of reference circles in size legend.
        marker_min_size: Minimum marker size used in the plot.
        marker_max_size: Maximum marker size used in the plot.
        effect_cap: Cap value used for effect magnitude.
        has_capped: Whether any values were capped.
        capped_marker_style: "circle" for uniform circles with "+" suffix, "diamond" for separate diamond entry.
    """
    # --- Line legend (tight spacing) ---
    line_elements = []
    if inf_proxy is not None:
        line_elements.append(Line2D([0], [0], color="#A0A0E8", lw=0.8, linestyle="dashdot", label="Infinity"))
    line_elements.extend([
        Line2D([0], [0], color="#82C882", lw=0.8, linestyle="dashed", label="Bonferroni\nCorrection"),
        Line2D([0], [0], color="#E8A0A0", lw=0.8, linestyle="dotted", label="Nominal\nSignificance"),
    ])

    line_legend = ax.legend(
        handles=line_elements,
        handlelength=2,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=legend_marker_size,
        labelspacing=1.0,
        frameon=False,
    )
    ax.add_artist(line_legend)

    # --- Size legend (wider spacing to fit circles) ---
    if show_size_legend and "_effect_magnitude" in plot_df.columns:
        mag_min = plot_df["_effect_magnitude"].min()
        mag_max = plot_df["_effect_magnitude"].max()

        if mag_min is not None and mag_max is not None and mag_max > mag_min:
            size_elements = []
            max_legend_s = marker_max_size

            ref_mags = np.round(np.linspace(mag_min, mag_max, size_legend_count), 1).tolist()
            ref_mags = list(dict.fromkeys(ref_mags))

            mag_range = mag_max - mag_min
            for idx, mag in enumerate(ref_mags):
                clamped = min(max(mag, mag_min), mag_max)
                legend_s = marker_min_size + (clamped - mag_min) * (marker_max_size - marker_min_size) / mag_range
                is_last = idx == len(ref_mags) - 1
                if capped_marker_style == "circle" and has_capped and is_last:
                    label = f"Effect={mag:.1f}+"
                else:
                    label = f"Effect={mag:.1f}"
                size_elements.append(
                    ax.scatter([], [], s=legend_s, c="gray", marker="o", label=label)
                )
                max_legend_s = max(max_legend_s, legend_s)

            if capped_marker_style == "diamond" and has_capped:
                size_elements.append(
                    ax.scatter([], [], s=marker_max_size, c="gray", marker="D",
                               label=f"Effect>{effect_cap:.0f}")
                )

            size_spacing = max(1.0, np.sqrt(max_legend_s) / legend_marker_size * 0.6)
            ax.legend(
                handles=size_elements,
                handlelength=2,
                loc="upper left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=legend_marker_size,
                labelspacing=size_spacing,
                frameon=False,
            )
