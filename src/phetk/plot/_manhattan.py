"""Manhattan plot backend."""
import adjustText
import matplotlib.colors as mc
import polars as pl
from matplotlib.lines import Line2D

from phetk.plot._base import PlotBackend, PlotContext, register
from phetk.plot._shared import (
    compute_effect_sizes,
    draw_lines,
    filter_by_phecode_categories,
    save_plot,
    split_by_effect_size,
    split_text,
    x_ticks,
)


@register("manhattan")
class ManhattanBackend(PlotBackend):
    """Backend for Manhattan plot visualization."""

    def render(self, ctx: PlotContext, **kwargs) -> None:
        """
        Create Manhattan plot visualization of PheWAS results.

        Args:
            ctx: Shared plot context with data and settings.
            **kwargs: All Manhattan plot parameters (see Plot.manhattan signature).
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
        marker_size_by_effect_size = kwargs.get("marker_size_by_effect_size", False)
        marker_scale_factor = kwargs.get("marker_scale_factor", 1)
        positive_marker_alpha = kwargs.get("positive_marker_alpha", 0.7)
        negative_marker_alpha = kwargs.get("negative_marker_alpha", 0.7)
        hide_non_significant = kwargs.get("hide_non_significant", False)
        phecode_categories = kwargs.get("phecode_categories", None)
        plot_all_categories = kwargs.get("plot_all_categories", True)
        sort_by_significance = kwargs.get("sort_by_significance", False)
        title = kwargs.get("title", None)
        title_text_size = kwargs.get("title_text_size", 10)
        y_limit = kwargs.get("y_limit", None)
        axis_text_size = kwargs.get("axis_text_size", 8)
        show_legend = kwargs.get("show_legend", True)
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
                    sort_by_significance
                )
            else:
                print("phecode_categories must not be None when plot_all_categories = False.")
                return

        # Compute effect sizes when requested
        if marker_size_by_effect_size:
            plot_df = compute_effect_sizes(plot_df, ctx.direction_col)

        # Optionally drop points below nominal significance
        if hide_non_significant:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") >= ctx.nominal_significance)

        # create plot
        ratio = n_categories / len(ctx.phewas_result.columns)
        fig, ax = adjustText.plt.subplots(figsize=(12 * ratio, 7), dpi=dpi)

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # filter data by y_limit if specified
        if y_limit is not None:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") <= y_limit)
            ax.set_ylim(-0.2, y_limit)

        # y axis label
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        # generate positive & negative betas
        positive_betas, negative_betas = split_by_effect_size(
            plot_df, ctx.direction_col
        )

        ############
        # PLOTTING #
        ############

        # x-axis offset
        adjustText.plt.xlim(float(plot_df["phecode_index"].min()) - offset - 1,
                            float(plot_df["phecode_index"].max()) + offset + 1)

        # create x ticks labels and colors
        x_ticks(plot_df, selected_color_dict)

        # scatter
        _manhattan_scatter(
            ax=ax,
            positive_betas=positive_betas,
            negative_betas=negative_betas,
            positive_alpha=positive_marker_alpha,
            negative_alpha=negative_marker_alpha,
            marker_size_by_effect_size=marker_size_by_effect_size,
            scale_factor=marker_scale_factor
        )

        # lines
        draw_lines(
            ax=ax,
            plot_type="manhattan",
            plot_df=plot_df,
            x_col="phecode_index",
            bonferroni=ctx.bonferroni,
            inf_proxy=ctx.inf_proxy,
            offset=offset,
            nominal_significance_line=True,
            bonferroni_line=True,
            infinity_line=True
        )

        # labeling
        _manhattan_label(
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
            label_box_alpha=label_box_alpha
        )

        # legend
        if show_legend:
            _manhattan_legend(ax, legend_marker_size, positive_marker_alpha,
                              negative_marker_alpha, ctx.inf_proxy)

        # save plot
        if do_save_plot:
            save_plot(plot_type="manhattan", output_file_path=output_file_path)


def _create_phecode_index(
        df: pl.DataFrame,
        sort_by_significance: bool = False
) -> pl.DataFrame:
    """
    Create sequential phecode index for Manhattan plot x-axis positioning.

    Args:
        df: PheWAS result dataframe to create index for.
        sort_by_significance: If True, sort by phecode_category and -log10(p-value) descending.

    Returns:
        Dataframe with phecode_index column added.
    """
    if "phecode_index" in df.columns:
        df = df.drop("phecode_index")

    if sort_by_significance:
        df = df.sort(by=["phecode_category", "neg_log_p_value"], descending=[False, True])
    else:
        df = df.sort(by=["phecode_category", "phecode"])

    df = df.with_columns(pl.Series("phecode_index", range(1, len(df) + 1)))
    return df


def _manhattan_scatter(
        ax,
        positive_betas: pl.DataFrame,
        negative_betas: pl.DataFrame,
        positive_alpha: float,
        negative_alpha: float,
        marker_size_by_effect_size: bool,
        scale_factor: float = 1
) -> None:
    """
    Generate scatter plot points for Manhattan plot.

    Args:
        ax: Matplotlib axes object for plotting.
        positive_betas: Dataframe of positive effect results.
        negative_betas: Dataframe of negative effect results.
        positive_alpha: Alpha for positive effect markers.
        negative_alpha: Alpha for negative effect markers.
        marker_size_by_effect_size: Whether to scale marker size by effect magnitude.
        scale_factor: Scaling factor for marker sizes.
    """
    if marker_size_by_effect_size:
        s_positive = positive_betas["_scaled_size"] * scale_factor
        s_negative = negative_betas["_scaled_size"] * scale_factor
    else:
        s_positive = None
        s_negative = None

    ax.scatter(
        x=positive_betas["phecode_index"].to_numpy(),
        y=positive_betas["neg_log_p_value"],
        s=s_positive,
        c=positive_betas["label_color"],
        marker="^",
        alpha=positive_alpha
    )

    ax.scatter(
        x=negative_betas["phecode_index"].to_numpy(),
        y=negative_betas["neg_log_p_value"],
        s=s_negative,
        c=negative_betas["label_color"],
        marker="v",
        alpha=negative_alpha
    )


def _manhattan_label(
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
        y_col: str = "neg_log_p_value",
        x_col: str = "phecode_index"
):
    """
    Add data point labels to Manhattan plot with automatic positioning.

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
        y_col: Column containing y-axis values.
        x_col: Column containing x-axis values.

    Returns:
        adjustText object for label positioning.
    """
    if isinstance(label_values, str):
        label_values = [label_values]

    data_to_label = pl.DataFrame(schema=plot_df.schema)
    pos = positive_betas.clone()
    neg = negative_betas.clone()

    for item in label_values:
        if item == "positive_beta":
            data_to_label = pl.concat(
                [data_to_label,
                 pos.filter(pl.col(direction_col) >= label_value_threshold)]
            )
            if label_categories is not None:
                data_to_label = data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                data_to_label = data_to_label[:label_count]
        elif item == "negative_beta":
            data_to_label = pl.concat(
                [data_to_label,
                 neg.filter(pl.col(direction_col) <= label_value_threshold)]
            )
            if label_categories is not None:
                data_to_label = data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                data_to_label = data_to_label[:label_count]
        elif item == "p_value":
            data_to_label = pl.concat(
                [data_to_label,
                 plot_df.sort(by="p_value")
                        .filter(pl.col("neg_log_p_value") >= label_value_threshold)]
            )
            if label_categories is not None:
                data_to_label = data_to_label.filter(
                    pl.col("phecode_category").is_in(label_categories)
                )[:label_count]
            else:
                data_to_label = data_to_label[:label_count]
        else:
            data_to_label = pl.concat(
                [data_to_label,
                 plot_df.filter(pl.col("phecode") == item)]
            )

    texts = []
    for i in range(len(data_to_label)):
        if mc.is_color_like(label_color):
            color = pl.Series(values=[label_color] * len(data_to_label))
        else:
            # noinspection PyTypeChecker
            color = data_to_label[label_color]
        # noinspection PyTypeChecker
        texts.append(
            adjustText.plt.text(
                float(data_to_label[x_col][i]),
                float(data_to_label[y_col][i]),
                split_text(
                    data_to_label[label_text_column][i],
                    label_split_threshold
                ),
                color=color[i],
                size=label_size,
                weight=label_weight,
                alpha=1,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    boxstyle="round",
                    alpha=label_box_alpha,
                    lw=0.5
                )
            )
        )

    if len(texts) > 0:
        return adjustText.adjust_text(
            texts,
            arrowprops=dict(
                arrowstyle="simple",
                color="gray", lw=0.5,
                mutation_scale=2
            )
        )


def _manhattan_legend(
        ax,
        legend_marker_size: int,
        positive_alpha: float,
        negative_alpha: float,
        inf_proxy: float | None
) -> None:
    """
    Create legend for Manhattan plot.

    Args:
        ax: Matplotlib axes object for legend.
        legend_marker_size: Size of markers in legend.
        positive_alpha: Alpha for positive effect markers.
        negative_alpha: Alpha for negative effect markers.
        inf_proxy: Infinity proxy value, or None if no infinity values.
    """
    legend_elements = []

    if inf_proxy is not None:
        legend_elements.append(Line2D([0], [0], color="#A0A0E8", lw=0.8, linestyle="dashdot", label="Infinity"))

    legend_elements.extend([
        Line2D([0], [0], color="#82C882", lw=0.8, linestyle="dashed", label="Bonferroni\nCorrection"),
        Line2D([0], [0], color="#E8A0A0", lw=0.8, linestyle="dotted", label="Nominal\nSignificance"),
        Line2D([0], [0], marker="^", label="Increased\nRisk Effect", color="white",
               markerfacecolor="blue", alpha=positive_alpha, markersize=legend_marker_size),
        Line2D([0], [0], marker="v", label="Decreased\nRisk Effect", color="white",
               markerfacecolor="blue", alpha=negative_alpha, markersize=legend_marker_size),
    ])
    ax.legend(
        handles=legend_elements,
        handlelength=2,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=legend_marker_size,
        frameon=False,
        labelspacing=1.2,
    )
