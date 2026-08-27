"""Forest plot backend."""
import matplotlib.pyplot as plt
import polars as pl

from phetk.plot._base import PlotBackend, PlotContext, register
from phetk.plot._shared import save_plot


@register("forest")
class ForestBackend(PlotBackend):
    """Backend for forest plot visualization."""

    def render(self, ctx: PlotContext, **kwargs) -> None:
        """
        Create forest plot for selected phecodes from PheWAS results.

        Args:
            ctx: Shared plot context with data and settings.
            **kwargs: All forest plot parameters (see Plot.forest signature).
        """
        # Extract kwargs
        phecode_list = kwargs.get("phecode_list", None)
        n_top_values = kwargs.get("n_top_values", 10)
        plot_odds_ratio = kwargs.get("plot_odds_ratio", True)
        show_phecode = kwargs.get("show_phecode", True)
        title = kwargs.get("title", None)
        axis_text_size = kwargs.get("axis_text_size", 10)
        label_size = kwargs.get("label_size", 10)
        marker_shape = kwargs.get("marker_shape", "s")
        marker_size = kwargs.get("marker_size", 6)
        highlight_significance = kwargs.get("highlight_significance", False)
        highlight_phecodes = kwargs.get("highlight_phecodes", None)
        highlight_p_value_threshold = kwargs.get("highlight_p_value_threshold", None)
        show_p_value_asterisks = kwargs.get("show_p_value_asterisks", False)
        show_count = kwargs.get("show_count", False)
        show_sex_restriction = kwargs.get("show_sex_restriction", False)
        dpi = kwargs.get("dpi", 150)
        do_save_plot = kwargs.get("save_plot", True)
        output_file_path = kwargs.get("output_file_path", None)

        # Auto-select top positive and negative effects if no phecode_list provided
        if phecode_list is None:
            # First detect the effect column to determine direction
            if "beta" in ctx.phewas_result.columns:
                effect_col = "beta"
            elif "hazard_ratio" in ctx.phewas_result.columns:
                effect_col = "hazard_ratio"
            elif "log_hazard_ratio" in ctx.phewas_result.columns:
                effect_col = "log_hazard_ratio"
            else:
                print("Could not find effect size column (beta, hazard_ratio, or log_hazard_ratio)")
                return

            # Get top positive and negative phecode lists
            if effect_col == "hazard_ratio":
                positive_phecodes = ctx.phewas_result.filter(pl.col(effect_col) > 1).top_k(
                    by=effect_col, k=n_top_values, reverse=False
                )["phecode"].to_list()
                negative_phecodes = ctx.phewas_result.filter(pl.col(effect_col) < 1).top_k(
                    by=effect_col, k=n_top_values, reverse=True
                )["phecode"].to_list()
            else:
                positive_phecodes = ctx.phewas_result.filter(pl.col(effect_col) > 0).top_k(
                    by=effect_col, k=n_top_values, reverse=False
                )["phecode"].to_list()
                negative_phecodes = ctx.phewas_result.filter(pl.col(effect_col) < 0).top_k(
                    by=effect_col, k=n_top_values, reverse=True
                )["phecode"].to_list()

            phecode_list = positive_phecodes + negative_phecodes

            if len(phecode_list) == 0:
                print("No data found with positive or negative effects.")
                return
        else:
            if isinstance(phecode_list, str):
                phecode_list = [phecode_list]

        # Filter data for specified phecodes
        plot_data = ctx.phewas_result.filter(pl.col("phecode").is_in(phecode_list))

        if len(plot_data) == 0:
            print("No data found for specified phecodes.")
            return

        # Detect effect size column
        if "beta" in plot_data.columns:
            if plot_odds_ratio:
                if "odds_ratio" in plot_data.columns:
                    effect_col = "odds_ratio"
                    effect_type = "Odds Ratio"
                else:
                    plot_data = plot_data.with_columns([
                        pl.col("beta").exp().alias("odds_ratio")
                    ])
                    effect_col = "odds_ratio"
                    effect_type = "Odds Ratio"
            else:
                effect_col = "beta"
                effect_type = "Beta"
        elif "hazard_ratio" in plot_data.columns:
            effect_col = "hazard_ratio"
            effect_type = "Hazard Ratio"
        else:
            print("Could not find effect size column (beta or hazard_ratio)")
            return

        # Auto-detect confidence interval columns
        if "conf_int_1" in plot_data.columns and "conf_int_2" in plot_data.columns:
            if effect_col == "odds_ratio":
                plot_data = plot_data.with_columns([
                    pl.col("conf_int_1").exp().alias("odds_ratio_ci_low"),
                    pl.col("conf_int_2").exp().alias("odds_ratio_ci_high")
                ])
                ci_cols = ("odds_ratio_ci_low", "odds_ratio_ci_high")
            else:
                ci_cols = ("conf_int_1", "conf_int_2")
        elif "hazard_ratio_low" in plot_data.columns and "hazard_ratio_high" in plot_data.columns:
            ci_cols = ("hazard_ratio_low", "hazard_ratio_high")
        elif "standard_error" in plot_data.columns:
            plot_data = plot_data.with_columns([
                (pl.col(effect_col) - 1.96 * pl.col("standard_error")).alias("ci_lower"),
                (pl.col(effect_col) + 1.96 * pl.col("standard_error")).alias("ci_upper")
            ])
            ci_cols = ("ci_lower", "ci_upper")
        else:
            print("Could not find confidence interval columns")
            return

        # Sort by effect size (largest effect first)
        plot_data = plot_data.sort(effect_col, descending=True)

        # Extract data for plotting
        effects = plot_data[effect_col].to_numpy()
        if effect_col == "hazard_ratio" or effect_col == "odds_ratio":
            reference_line = 1.0
        else:
            reference_line = 0.0
        x_label = effect_type

        ci_lows = plot_data[ci_cols[0]].to_numpy()
        ci_highs = plot_data[ci_cols[1]].to_numpy()
        p_values = plot_data["p_value"].to_numpy()
        phecodes = plot_data["phecode"].to_list()
        if show_phecode:
            phecode_strings = [f"{string} ({phecode})" for string, phecode
                               in zip(plot_data["phecode_string"].to_list(), phecodes)]
        else:
            phecode_strings = plot_data["phecode_string"].to_list()

        # Process highlight_phecodes parameter
        if highlight_phecodes is not None:
            if isinstance(highlight_phecodes, str):
                highlight_phecodes = [highlight_phecodes]
            highlight_phecodes_set = set(highlight_phecodes)
        else:
            highlight_phecodes_set = set()

        # Set p-value threshold
        if highlight_p_value_threshold is None:
            highlight_p_value_threshold = 10 ** (-ctx.bonferroni)

        # Create figure with subplots
        n_phecodes = len(plot_data)

        panels = 4  # Base panels: text, forest, effect, p-value
        width_ratios = [2, 4, 1.5, 0.8]

        if show_count:
            panels += 1
            width_ratios.append(1.8)

        if show_sex_restriction:
            panels += 1
            width_ratios.append(0.4)

        calculated_figsize = (sum(width_ratios) * 1.5, n_phecodes * 0.4)

        fig, axes = plt.subplots(
            1, panels,
            figsize=calculated_figsize,
            gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05},
            dpi=dpi
        )

        # Assign axes based on panels
        ax_text = axes[0]
        ax_forest = axes[1]
        ax_effect = axes[2]
        ax_pval = axes[3]

        if show_count and show_sex_restriction:
            ax_count = axes[4]
            ax_sex = axes[5]
        elif show_count:
            ax_count = axes[4]
            ax_sex = None
        elif show_sex_restriction:
            ax_count = None
            ax_sex = axes[4]
        else:
            ax_count = None
            ax_sex = None

        # Forest plot (second panel)
        if title is not None:
            ax_forest.set_title(title, fontweight='bold', fontsize=axis_text_size + 2)

        # Plot confidence intervals as horizontal lines
        for i, (effect, ci_low, ci_high, pval, phecode) in enumerate(
                zip(effects, ci_lows, ci_highs, p_values, phecodes)):
            should_highlight = ((highlight_significance and pval <= highlight_p_value_threshold)
                                or (phecode in highlight_phecodes_set))
            if should_highlight:
                line_width = 2
                marker_edge_width = 1.5
            else:
                line_width = 1
                marker_edge_width = 1

            # Confidence interval line
            ax_forest.plot([ci_low, ci_high], [i, i], 'k-', linewidth=line_width, alpha=0.7)

            # Point estimate
            color = _get_marker_color(effect, effect_col)

            ax_forest.plot(effect, i, marker_shape, color=color, markersize=marker_size,
                           markeredgecolor='black', linewidth=marker_edge_width)

            # Add caps to confidence interval
            cap_height = 0.1
            ax_forest.plot([ci_low, ci_low], [i - cap_height, i + cap_height], 'k-', linewidth=line_width)
            ax_forest.plot([ci_high, ci_high], [i - cap_height, i + cap_height], 'k-', linewidth=line_width)

        # Add reference line
        ax_forest.axvline(x=reference_line, color='black', linestyle='--', alpha=0.5)

        # Format forest plot
        ax_forest.set_xlabel(x_label, fontweight='bold', fontsize=axis_text_size)
        ax_forest.set_yticks([])
        ax_forest.set_ylim(-0.5, n_phecodes - 0.5)
        ax_forest.grid(False)
        ax_forest.invert_yaxis()

        ax_forest.spines['top'].set_visible(False)
        ax_forest.spines['right'].set_visible(False)
        ax_forest.spines['left'].set_visible(False)

        # Text panel (first panel) - Phecode descriptions
        panel_title = 'Phenotype (phecode)   ' if show_phecode else 'Phenotype   '
        ax_text.set_title(panel_title, fontweight='bold', fontsize=axis_text_size, loc="right")
        _setup_panel_and_add_text(
            ax_text, '', phecode_strings, p_values, phecodes, effects,
            highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
            effect_col, n_phecodes, axis_text_size, label_size, ha='right'
        )

        # Effect (CI) panel (third panel)
        effect_texts = []
        for effect, ci_low, ci_high in zip(effects, ci_lows, ci_highs):
            effect_text = f"{effect:.3f} ({ci_low:.3f}, {ci_high:.3f})"
            effect_texts.append(effect_text)

        _setup_panel_and_add_text(ax_effect, f'{effect_type} (95% CI)', effect_texts, p_values, phecodes, effects,
                                  highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                  effect_col, n_phecodes, axis_text_size, label_size)

        # p-value panel (fourth panel)
        pval_texts = []
        for pval in p_values:
            pval_text = f"{pval:.2e}"
            if show_p_value_asterisks:
                asterisks = _get_p_value_asterisks(pval)
                pval_text += asterisks
            pval_texts.append(pval_text)

        _setup_panel_and_add_text(ax_pval, 'p-value', pval_texts, p_values, phecodes, effects,
                                  highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                  effect_col, n_phecodes, axis_text_size, label_size)

        # Cases/Controls panel
        if show_count:
            count_texts = []
            for cases, controls in zip(plot_data["cases"], plot_data["controls"]):
                total_n = cases + controls
                count_text = f"{total_n:,}({cases:,}/{controls:,})"
                count_texts.append(count_text)

            _setup_panel_and_add_text(ax_count, 'N(cases/controls)', count_texts, p_values, phecodes, effects,
                                      highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                      effect_col, n_phecodes, axis_text_size, label_size)

        # Sex restriction panel
        if show_sex_restriction:
            _setup_panel_and_add_text(ax_sex, 'Sex', plot_data["phecode_sex_restriction"], p_values, phecodes, effects,
                                      highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                      effect_col, n_phecodes, axis_text_size, label_size)

        # Save plot
        if do_save_plot:
            save_plot(plot_type="forest", output_file_path=output_file_path)


def _get_marker_color(effect_val: float, effect_column: str,
                      positive_color: str = '#D55E00', negative_color: str = '#56B4E9') -> str:
    """Determine marker color based on effect size and regression type."""
    if effect_column == "hazard_ratio" or effect_column == "odds_ratio":
        return positive_color if effect_val > 1.0 else negative_color
    else:
        return positive_color if effect_val > 0.0 else negative_color


def _get_text_color(effect_val: float, effect_column: str, highlight: bool) -> str:
    """Determine text color based on highlighting and effect direction."""
    if highlight:
        return _get_marker_color(effect_val, effect_column)
    else:
        return 'black'


def _get_p_value_asterisks(p_val: float) -> str:
    """Generate significance asterisks based on p-value thresholds."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


def _setup_panel_and_add_text(
        ax, title: str, texts, p_values, phecodes, effects,
        highlight_significance: bool, highlight_p_value_threshold: float,
        highlight_phecodes_set: set, effect_col: str, n_phecodes: int,
        axis_text_size: int, label_size: int, ha: str = 'center'
):
    """Setup panel formatting and add text with highlighting."""
    ax.set_title(title, fontweight='bold', fontsize=axis_text_size)

    for i, (text, pval, phecode, effect) in enumerate(zip(texts, p_values, phecodes, effects)):
        should_highlight = ((highlight_significance and pval <= highlight_p_value_threshold)
                            or (phecode in highlight_phecodes_set))
        weight = 'bold' if should_highlight else 'normal'
        text_color = _get_text_color(effect, effect_col, should_highlight)

        ax.text(0.5 if ha == 'center' else 0.95, i, text, va='center', ha=ha,
                fontsize=label_size, weight=weight, color=text_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_phecodes - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)
