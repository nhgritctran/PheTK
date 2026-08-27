"""Volcano plot backend."""
import adjustText
import polars as pl

from phetk.plot._base import PlotBackend, PlotContext, register
from phetk.plot._shared import (
    draw_lines,
    save_plot,
    split_by_effect_size,
    split_text,
    transform_values,
)


@register("volcano")
class VolcanoBackend(PlotBackend):
    """Backend for volcano plot visualization."""

    def render(self, ctx: PlotContext, **kwargs) -> None:
        """
        Create volcano plot visualization of PheWAS results.

        Args:
            ctx: Shared plot context with data and settings.
            **kwargs: All volcano plot parameters (see Plot.volcano signature).
        """
        # Extract kwargs
        phecode_list = kwargs.get("phecode_list", None)
        phecode_string_list = kwargs.get("phecode_string_list", None)
        label_count = kwargs.get("label_count", 10)
        x_col = kwargs.get("x_col", "log10_odds_ratio")
        y_col = kwargs.get("y_col", "neg_log_p_value")
        x_axis_label = kwargs.get("x_axis_label", None)
        exclude_infinity = kwargs.get("exclude_infinity", False)
        y_threshold = kwargs.get("y_threshold", None)
        x_negative_threshold = kwargs.get("x_negative_threshold", None)
        x_positive_threshold = kwargs.get("x_positive_threshold", None)
        bonferroni_line = kwargs.get("bonferroni_line", False)
        nominal_significance_line = kwargs.get("nominal_significance_line", False)
        infinity_line = kwargs.get("infinity_line", False)
        y_limit = kwargs.get("y_limit", None)
        title = kwargs.get("title", None)
        title_text_size = kwargs.get("title_text_size", None)
        axis_text_size = kwargs.get("axis_text_size", None)
        marker_size_col = kwargs.get("marker_size_col", "cases")
        marker_shape = kwargs.get("marker_shape", ".")
        fill_marker = kwargs.get("fill_marker", True)
        marker_alpha = kwargs.get("marker_alpha", 0.5)
        label_box_alpha = kwargs.get("label_box_alpha", 0.5)
        show_legend = kwargs.get("show_legend", False)
        legend_marker_scale = kwargs.get("legend_marker_scale", 0.5)
        legend_label_count = kwargs.get("legend_label_count", 5)
        dpi = kwargs.get("dpi", 150)
        do_save_plot = kwargs.get("save_plot", True)
        output_file_path = kwargs.get("output_file_path", None)

        # Local state
        offset = 0.1
        ratio = 1

        # create plot
        fig, ax = adjustText.plt.subplots(figsize=(12 * ratio, 7), dpi=dpi)

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # plot_df
        plot_df = ctx.phewas_result.clone()
        if exclude_infinity:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") != ctx.inf_proxy)

        # filter data by y_limit if specified
        if y_limit is not None:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") <= y_limit)
            ax.set_ylim(-0.2, y_limit)

        # x, y axis label
        if x_col == "log10_odds_ratio":
            x_axis_label = r"$\log_{10}$(OR)"
        elif (x_col != "log10_odds_ratio") and (x_axis_label is None):
            x_axis_label = x_col
        ax.set_xlabel(x_axis_label, size=axis_text_size)
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        # generate positive & negative betas
        positive_betas, negative_betas = split_by_effect_size(plot_df, ctx.direction_col)

        ############
        # PLOTTING #
        ############

        # scatter
        _volcano_scatter(
            ax=ax,
            positive_betas=positive_betas,
            negative_betas=negative_betas,
            x_col=x_col,
            marker_size_col=marker_size_col,
            marker_shape=marker_shape,
            fill_marker=fill_marker,
            marker_alpha=marker_alpha,
            legend_marker_scale=legend_marker_scale,
            legend_label_count=legend_label_count,
            show_legend=show_legend
        )

        # lines
        x_positive_threshold_line = x_positive_threshold is not None
        x_negative_threshold_line = x_negative_threshold is not None
        y_threshold_line = y_threshold is not None
        draw_lines(
            ax=ax,
            plot_type="volcano",
            plot_df=plot_df,
            x_col=x_col,
            bonferroni=ctx.bonferroni,
            inf_proxy=ctx.inf_proxy,
            offset=offset,
            y_threshold_line=y_threshold_line,
            y_threshold_value=y_threshold,
            x_positive_threshold_line=x_positive_threshold_line,
            x_positive_threshold_value=x_positive_threshold,
            x_negative_threshold_line=x_negative_threshold_line,
            x_negative_threshold_value=x_negative_threshold,
            bonferroni_line=bonferroni_line,
            nominal_significance_line=nominal_significance_line,
            infinity_line=infinity_line
        )

        # labels
        _volcano_label(
            positive_betas=positive_betas,
            negative_betas=negative_betas,
            phecode_list=phecode_list,
            phecode_string_list=phecode_string_list,
            plot_df=plot_df,
            label_count=label_count,
            x_col=x_col,
            y_col=y_col,
            label_box_alpha=label_box_alpha,
            y_threshold=y_threshold,
            x_positive_threshold=x_positive_threshold,
            x_negative_threshold=x_negative_threshold
        )

        # save plot
        if do_save_plot:
            save_plot(plot_type="volcano", output_file_path=output_file_path)


def _volcano_scatter(
        ax,
        positive_betas: pl.DataFrame,
        negative_betas: pl.DataFrame,
        x_col: str = "log10_odds_ratio",
        y_col: str = "neg_log_p_value",
        marker_size_col: str | None = "cases",
        marker_shape: str = ".",
        positive_beta_color: str = "indianred",
        negative_beta_color: str = "darkcyan",
        fill_marker: bool = True,
        marker_alpha: float = 0.5,
        legend_marker_scale: float = 0.5,
        legend_label_count: int = 5,
        show_legend: bool = False
) -> None:
    """
    Create scatter plot for volcano plot visualization.

    Args:
        ax: Matplotlib axes object for plotting.
        positive_betas: Positive effect dataframe.
        negative_betas: Negative effect dataframe.
        x_col: Column name for x-axis values (effect size).
        y_col: Column name for y-axis values (significance).
        marker_size_col: Column name for marker sizing, constant size if None.
        marker_shape: Shape of markers for plotting.
        positive_beta_color: Color for positive effect markers.
        negative_beta_color: Color for negative effect markers.
        fill_marker: Whether to fill markers with color.
        marker_alpha: Transparency level for markers.
        legend_marker_scale: Scale factor for legend markers.
        legend_label_count: Number of items in size legend.
        show_legend: Whether to display size legend.
    """
    # set marker edge and face colors
    if fill_marker:
        positive_face_color = positive_beta_color
        negative_face_color = negative_beta_color
    else:
        positive_face_color = "none"
        negative_face_color = "none"

    # color values for every point
    if marker_size_col is not None:
        col_list = [x_col, y_col, marker_size_col]
    else:
        col_list = [x_col, y_col]
    pos_df = positive_betas[col_list].with_columns(pl.lit(positive_beta_color)
                                                    .alias("edge_color")) \
                                      .with_columns(pl.lit(positive_face_color)
                                                    .alias("face_color"))
    neg_df = negative_betas[col_list].with_columns(pl.lit(negative_beta_color)
                                                    .alias("edge_color")) \
                                      .with_columns(pl.lit(negative_face_color)
                                                    .alias("face_color"))
    # combined into 1 df for plotting
    full_df = pl.concat([pos_df, neg_df]).unique()
    if marker_size_col is not None:
        full_df = transform_values(df=full_df,
                                   col=marker_size_col,
                                   new_col="_marker_size",
                                   new_min=50,
                                   new_max=1000)
        marker_size = full_df["_marker_size"].to_numpy()
    else:
        marker_size = None

    # plot scatter
    scatter = ax.scatter(
        x=full_df[x_col].to_numpy(),
        y=full_df[y_col],
        s=marker_size,
        edgecolors=full_df["edge_color"],
        facecolors=full_df["face_color"],
        marker=marker_shape,
        alpha=marker_alpha,
    )

    # legend
    k = 0.05
    min_size = full_df[marker_size_col].min()
    max_size = full_df[marker_size_col].max()
    margin = (max_size - min_size) * k
    step_size = (max_size - min_size) * (1 - (2 * k)) / (legend_label_count - 1)
    legend_labels = [
        min_size + margin + (i * step_size) for i in range(legend_label_count)
    ]
    legend_labels = [round(i, -2) for i in legend_labels]
    if (marker_size_col is not None) and show_legend:
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, num=legend_label_count)
        ax.legend(
            handles=handles,
            labels=legend_labels,
            markerscale=legend_marker_scale,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=marker_size_col,
            frameon=False,
        )


def _volcano_label(
        positive_betas: pl.DataFrame,
        negative_betas: pl.DataFrame,
        plot_df: pl.DataFrame,
        phecode_list: list[str] | str | None = None,
        phecode_string_list: list[str] | str | None = None,
        x_col: str = "log10_odds_ratio",
        y_col: str = "neg_log_p_value",
        label_count: int = 10,
        label_text_column: str = "phecode_string",
        label_split_threshold: int = 30,
        label_size: int = 8,
        label_weight: str = "normal",
        label_box_alpha: float = 0.5,
        y_threshold: float | None = 5,
        x_positive_threshold: float | None = None,
        x_negative_threshold: float | None = None
):
    """
    Add data point labels to volcano plot with automatic positioning.

    Args:
        positive_betas: Positive effect dataframe.
        negative_betas: Negative effect dataframe.
        plot_df: Plot dataframe containing results to label.
        phecode_list: Specific phecodes to label.
        phecode_string_list: Specific phecode descriptions to label.
        x_col: Column name for x-axis values.
        y_col: Column name for y-axis values.
        label_count: Maximum number of labels to display.
        label_text_column: Column containing text for labels.
        label_split_threshold: Character threshold for splitting long labels.
        label_size: Font size for labels.
        label_weight: Font weight for labels.
        label_box_alpha: Alpha value for label background boxes.
        y_threshold: Minimum significance threshold for labeling.
        x_positive_threshold: Minimum positive effect threshold for labeling.
        x_negative_threshold: Maximum negative effect threshold for labeling.

    Returns:
        adjustText object for label positioning.
    """
    # Get the data for labeling
    if (phecode_list is not None) or (phecode_string_list is not None):
        if isinstance(phecode_string_list, str):
            phecode_string_list = [phecode_string_list]
        if isinstance(phecode_list, str):
            phecode_list = [phecode_list]
        data_to_label = plot_df.filter((pl.col("phecode").is_in(phecode_list)) |
                                       (pl.col("phecode_string").is_in(phecode_string_list)))
    elif (y_threshold is not None) or (x_negative_threshold is not None) or (x_positive_threshold is not None):
        if y_threshold is None:
            y_threshold = plot_df["neg_log_p_value"].min()
        if x_negative_threshold is None:
            x_negative_threshold = plot_df[x_col].max()
        if x_positive_threshold is None:
            x_positive_threshold = plot_df[x_col].min()
        data_to_label = plot_df.filter(
            ((pl.col(x_col) >= x_positive_threshold) | (pl.col(x_col) <= x_negative_threshold)) &
            (pl.col("neg_log_p_value") >= y_threshold)
        )
        data_to_label = pl.concat(
            [data_to_label.top_k(round(label_count // 2), by=x_col),
             data_to_label.top_k(round(label_count // 2), by=x_col, reverse=True)]
        ).unique()
    else:
        data_to_label = pl.concat(
            [plot_df.top_k(round(label_count // 2), by=x_col),
             plot_df.top_k(round(label_count // 2), by=x_col, reverse=True)]
        ).unique()

    texts = []
    for i in range(len(data_to_label)):
        if data_to_label[x_col][i] < 0:
            color = "green"
        else:
            color = "red"
        # noinspection PyTypeChecker
        texts.append(adjustText.plt.text(float(data_to_label[x_col][i]),
                                         float(data_to_label[y_col][i]),
                                         split_text(data_to_label[label_text_column][i],
                                                    label_split_threshold),
                                         color=color,
                                         size=label_size,
                                         weight=label_weight,
                                         bbox=dict(facecolor="white",
                                                   edgecolor="none",
                                                   boxstyle="round",
                                                   alpha=label_box_alpha,
                                                   lw=0.5),
                                         alpha=1))
    if len(texts) > 0:
        return adjustText.adjust_text(
            texts, arrowprops=dict(arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2)
        )
