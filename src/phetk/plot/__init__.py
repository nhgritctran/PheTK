"""PheWAS plot backends. Importing registers all built-in backends."""
import matplotlib.colors as mc
import numpy as np
import polars as pl
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _utils
from phetk.plot._base import PlotBackend, PlotContext, available_plot_types, get_plot_backend
from phetk.plot._shared import (
    MARKER_MAX_SIZE,
    MARKER_MIN_SIZE,
    save_plot as _save_plot,
    transform_values as _transform_values,
)
from phetk.plot import _manhattan, _volcano, _forest, _miami  # noqa: F401 — trigger registration

__all__ = ["Plot", "PlotBackend", "PlotContext", "available_plot_types", "get_plot_backend"]


class Plot:
    def __init__(
            self,
            phewas_result_file_path: str,
            converged_only: bool = True,
            bonferroni: float | None = None,
            phecode_version: str | None = None,
            color_palette: tuple[str, ...] | str | None = None
    ):
        """
        Initialize Plot object for creating PheWAS visualization plots.

        Loads PheWAS results, configures plotting parameters, assigns colors to
        phecode categories, and prepares data for Manhattan and volcano plots.

        Args:
            phewas_result_file_path: Path to PheWAS result CSV/TSV file generated from PheWAS module.
            converged_only: Whether to plot converged results only.
            bonferroni: Bonferroni correction threshold, calculated based on number of phecodes tested if None.
            phecode_version: Phecode version ("1.2" or "X"), defaults to "X" if None.
            color_palette: Color palette - "default", "colorblind", "rainbow", or custom tuple of colors.
        """

        # load PheWAS results
        # "converged" is read as text and normalized below rather than parsed as Boolean, so
        # that result files written by any PheTK version load; see _normalize_converged
        sep = _utils.detect_delimiter(phewas_result_file_path)
        self.phewas_result = pl.read_csv(
            phewas_result_file_path,
            separator=sep,
            schema_overrides={"phecode": str, "converged": pl.Utf8}
        )
        self.phewas_result = self._normalize_converged(self.phewas_result)

        # bonferroni
        if bonferroni is None:
            self.bonferroni = -np.log10(0.05 / len(self.phewas_result))
        else:
            self.bonferroni = bonferroni

        # remove non-converged phecodes - doing this after bonferroni to avoid bonferroni value shifting
        if ("converged" in self.phewas_result.columns) and converged_only:
            self.phewas_result = self.phewas_result.filter(pl.col("converged"))

        # drop rows with NaN p_value (unreliable inference)
        nan_p_count = self.phewas_result.filter(pl.col("p_value").is_nan()).height
        if nan_p_count > 0:
            print(f"Plot: {nan_p_count} phecode(s) removed with NaN p-value (unreliable inference).")
            self.phewas_result = self.phewas_result.filter(pl.col("p_value").is_not_nan())

        # assign a proxy value for infinity neg_log_p_value
        max_non_inf_neg_log = self.phewas_result.filter(pl.col("neg_log_p_value") != np.inf) \
            .sort(by="neg_log_p_value", descending=True)["neg_log_p_value"][0]
        if max_non_inf_neg_log < self.phewas_result["neg_log_p_value"].max():
            self.inf_proxy = max_non_inf_neg_log * 1.2
            self.phewas_result = self.phewas_result.with_columns(pl.when(pl.col("neg_log_p_value") == np.inf)
                                                                 .then(self.inf_proxy)
                                                                 .otherwise(pl.col("neg_log_p_value"))
                                                                 .alias("neg_log_p_value"))
        else:
            self.inf_proxy = None

        # nominal significance
        self.nominal_significance = -np.log10(0.05)

        # phecode_version
        if phecode_version is not None:
            self.phecode_version = phecode_version.upper()
        else:
            self.phecode_version = "X"

        # phecode categories
        self.phecode_categories = None

        # color mapping
        if color_palette is None or color_palette == "default" or color_palette == "phetk":
            # 18-color palette: colorblind-friendly, alternating dark/light
            # so categories remain distinguishable even in grayscale print.
            # Dark entries (odd positions) have low luminance; light entries
            # (even positions) have high luminance — adjacent categories
            # always contrast in brightness regardless of hue perception.
            self.color_palette = (
                "#264F8A",  # 1  dark blue
                "#F5A623",  # 2  light amber
                "#1A7847",  # 3  dark forest green
                "#E8818A",  # 4  light salmon
                "#5B2C83",  # 5  dark purple
                "#6EC4CF",  # 6  light teal
                "#B84525",  # 7  dark vermilion
                "#A8D865",  # 8  light chartreuse
                "#862E5A",  # 9  dark magenta
                "#F4CE63",  # 10 light gold
                "#2A6B63",  # 11 dark teal
                "#C19BD6",  # 12 light lavender
                "#7A5420",  # 13 dark ochre
                "#6CB4EE",  # 14 light sky blue
                "#9E3040",  # 15 dark crimson
                "#85D9A8",  # 16 light mint
                "#3B3B6D",  # 17 dark indigo
                "#E8B89D",  # 18 light peach
            )
        elif color_palette == "classic":
            self.color_palette = (
                "blue", "indianred", "darkcyan", "goldenrod", "darkblue",
                "magenta", "green", "red", "darkturquoise", "olive",
                "black", "royalblue", "maroon", "darkolivegreen", "coral",
                "purple", "gray"
            )
        elif color_palette == "colorblind":
            # Color blind friendly palette based on Wong (2011) and other accessible colors
            self.color_palette = (
                "#0173B2",  # blue
                "#DE8F05",  # orange
                "#029E73",  # green
                "#CC78BC",  # light purple
                "#CA9161",  # light brown
                "#FBAFE4",  # light pink
                "#949494",  # gray
                "#ECE133",  # yellow
                "#56B4E9",  # sky blue
                "#E69F00",  # orange yellow
                "#0072B2",  # dark blue
                "#D55E00",  # vermilion
                "#CC79A7",  # reddish purple
                "#999999",  # medium gray
                "#F0E442",  # light yellow
                "#009E73",  # bluish green
                "#000000"   # black
            )
        elif color_palette == "rainbow":
            # Rainbow gradient with 17 distinct colors
            import matplotlib.cm as cm
            rainbow_colors = cm.rainbow(np.linspace(0, 1, 17))
            self.color_palette = tuple([mc.to_hex(color) for color in rainbow_colors])
        elif isinstance(color_palette, (tuple, list)):
            self.color_palette = tuple(color_palette)
        else:
            raise ValueError(f"Invalid color_palette: {color_palette}. Use 'default', 'classic', 'colorblind', 'rainbow', or a tuple of colors.")

        self.phecode_categories = self.phewas_result["phecode_category"].unique().to_list()
        self.phecode_categories.sort()
        self.color_dict = {self.phecode_categories[i]: self.color_palette[i % len(self.color_palette)]
                           for i in range(len(self.phecode_categories))}
        self.phewas_result = self.phewas_result.with_columns(
            pl.col("phecode_category").replace(self.color_dict).alias("label_color")
        )

        # column name for the datapoint direction
        self.direction_col = None
        if "beta" in self.phewas_result.columns:
            self.direction_col = "beta"
        elif "log_hazard_ratio" in self.phewas_result.columns:
            self.direction_col = "log_hazard_ratio"

        # Build context for backends
        self._ctx = PlotContext(
            phewas_result=self.phewas_result,
            bonferroni=self.bonferroni,
            nominal_significance=self.nominal_significance,
            phecode_version=self.phecode_version,
            phecode_categories=self.phecode_categories,
            color_dict=self.color_dict,
            color_palette=self.color_palette,
            inf_proxy=self.inf_proxy,
            direction_col=self.direction_col,
        )

    @staticmethod
    def _normalize_converged(df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize the "converged" column to Boolean, accepting all historical spellings.

        PheTK has written this column three ways: real booleans since v0.3.5, "True"/"False"
        from statsmodels, and "Converged"/"Not converged" from the Firth backends up to
        v0.3.4. The column is read as text so that every one of these loads, then mapped
        here. Values outside the known vocabulary become null and are treated as
        non-converged by converged_only.

        Args:
            df: PheWAS results as read from file.

        Returns:
            The frame with "converged" cast to Boolean, or unchanged if the column is absent.
        """
        if "converged" not in df.columns:
            return df

        normalized = pl.col("converged").str.strip_chars().str.to_lowercase()
        return df.with_columns(
            pl.when(normalized.is_in(["true", "converged"])).then(True)
            .when(normalized.is_in(["false", "not converged"])).then(False)
            .otherwise(None)
            .alias("converged")
        )

    @staticmethod
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
        _save_plot(plot_type=plot_type, output_file_path=output_file_path)

    @staticmethod
    def transform_values(
            df: pl.DataFrame,
            col: str,
            new_col: str,
            new_min: float,
            new_max: float
    ) -> pl.DataFrame:
        """
        Transform column values to specified range using min-max normalization.

        Scales values in specified column to new range while preserving
        relative relationships between data points.

        Args:
            df: Input dataframe containing column to transform.
            col: Name of column to transform.
            new_col: Name for new column containing transformed values.
            new_min: Minimum value for transformed range.
            new_max: Maximum value for transformed range.

        Returns:
            Dataframe with additional column containing transformed values.
        """
        return _transform_values(df=df, col=col, new_col=new_col, new_min=new_min, new_max=new_max)

    def manhattan(
            self,
            label_values: str | list[str] = "p_value",
            label_value_threshold: float = 0,
            label_count: int = 10,
            label_size: int = 8,
            label_text_column: str = "phecode_string",
            label_color: str = "label_color",
            label_weight: str = "normal",
            label_split_threshold: int = 30,
            label_box_alpha: float = 0.5,
            marker_size_by_effect_size: bool = False,
            marker_scale_factor: float = 1,
            positive_marker_alpha: float = 0.7,
            negative_marker_alpha: float = 0.7,
            hide_non_significant: bool = False,
            phecode_categories: list[str] | str | None = None,
            plot_all_categories: bool = True,
            sort_by_significance: bool = False,
            title: str | None = None,
            title_text_size: int = 10,
            y_limit: float | None = None,
            axis_text_size: int = 8,
            show_legend: bool = True,
            legend_marker_size: int = 7,
            dpi: int = 150,
            save_plot: bool = True,
            output_file_path: str | None = None
    ) -> None:
        """
        Create Manhattan plot visualization of PheWAS results.

        Generates comprehensive Manhattan plot showing -log10(p-values) across
        phecode categories with customizable labeling, significance lines,
        and effect direction indicators.

        Args:
            label_values: Criteria for labeling points - specific phecodes, "positive_beta", "negative_beta", or "p_value".
            label_value_threshold: Threshold for filtering labels by effect size or p-value.
            label_count: Maximum number of points to label.
            label_size: Font size for data point labels.
            label_text_column: Column containing text for labels.
            label_color: Color specification or column name for label colors.
            label_weight: Font weight for labels.
            label_split_threshold: Character threshold for splitting long labels.
            label_box_alpha: Alpha (transparency) value for label background boxes.
            marker_size_by_effect_size: Whether to scale marker size by effect magnitude.
            marker_scale_factor: Scaling factor for marker sizes.
            positive_marker_alpha: Alpha (transparency) value for positive effect markers.
            negative_marker_alpha: Alpha (transparency) value for negative effect markers.
            hide_non_significant: If True, omit points below nominal significance (p > 0.05).
            phecode_categories: Specific categories to plot, uses all if None.
            plot_all_categories: Whether to include all categories in plot.
            sort_by_significance: If True, sort phecodes by significance within each category.
            title: Plot title text.
            title_text_size: Font size for plot title.
            y_limit: Maximum y-axis value for display.
            axis_text_size: Font size for axis labels.
            show_legend: Whether to display plot legend.
            legend_marker_size: Size of markers in legend.
            dpi: Plot resolution in dots per inch.
            save_plot: Whether to save plot to file.
            output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), auto-generated if None.
        """
        get_plot_backend("manhattan").render(self._ctx, **{k: v for k, v in locals().items() if k != "self"})

    def volcano(
            self,
            phecode_list: list[str] | str | None = None,
            phecode_string_list: list[str] | str | None = None,
            label_count: int = 10,
            x_col: str = "log10_odds_ratio",
            y_col: str = "neg_log_p_value",
            x_axis_label: str | None = None,
            exclude_infinity: bool = False,
            y_threshold: float | None = None,
            x_negative_threshold: float | None = None,
            x_positive_threshold: float | None = None,
            bonferroni_line: bool = False,
            nominal_significance_line: bool = False,
            infinity_line: bool = False,
            y_limit: float | None = None,
            title: str | None = None,
            title_text_size: int | None = None,
            axis_text_size: int | None = None,
            marker_size_col: str | None = "cases",
            marker_shape: str = ".",
            fill_marker: bool = True,
            marker_alpha: float = 0.5,
            label_box_alpha: float = 0.5,
            show_legend: bool = False,
            legend_marker_scale: float = 0.5,
            legend_label_count: int = 5,
            dpi: int = 150,
            save_plot: bool = True,
            output_file_path: str | None = None
    ) -> None:
        """
        Create volcano plot visualization of PheWAS results.

        Generates volcano plot showing effect size vs significance with
        customizable thresholds, labeling, and visual elements.

        Args:
            phecode_list: Specific phecodes to label on plot.
            phecode_string_list: Specific phecode descriptions to label.
            label_count: Maximum number of points to label.
            x_col: Column name for x-axis values (effect size).
            y_col: Column name for y-axis values (significance).
            x_axis_label: Custom label for x-axis.
            exclude_infinity: Whether to exclude infinite significance values.
            y_threshold: Significance threshold for labeling and reference line.
            x_negative_threshold: Negative effect threshold for labeling and reference line.
            x_positive_threshold: Positive effect threshold for labeling and reference line.
            bonferroni_line: Whether to display Bonferroni correction line.
            nominal_significance_line: Whether to display nominal significance line.
            infinity_line: Whether to display infinity proxy line.
            y_limit: Maximum y-axis value for display.
            title: Plot title text.
            title_text_size: Font size for plot title.
            axis_text_size: Font size for axis labels.
            marker_size_col: Column name for marker sizing.
            marker_shape: Shape of markers for plotting.
            fill_marker: Whether to fill markers with color.
            marker_alpha: Transparency level for markers.
            label_box_alpha: Alpha (transparency) value for label background boxes.
            show_legend: Whether to display size legend.
            legend_marker_scale: Scale factor for legend markers.
            legend_label_count: Number of items in size legend.
            dpi: Plot resolution in dots per inch.
            save_plot: Whether to save plot to file.
            output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), auto-generated if None.
        """
        get_plot_backend("volcano").render(self._ctx, **{k: v for k, v in locals().items() if k != "self"})

    def forest(
            self,
            phecode_list: list[str] | str | None = None,
            n_top_values: int = 10,
            plot_odds_ratio: bool = True,
            show_phecode: bool = True,
            title: str | None = None,
            axis_text_size: int = 10,
            label_size: int = 10,
            marker_shape: str = "s",
            marker_size: int = 6,
            highlight_significance: bool = False,
            highlight_phecodes: list[str] | str | None = None,
            highlight_p_value_threshold: float | None = None,
            show_p_value_asterisks: bool = False,
            show_count: bool = False,
            show_sex_restriction: bool = False,
            dpi: int = 150,
            save_plot: bool = True,
            output_file_path: str | None = None
    ) -> None:
        """
        Create forest plot for selected phecodes from PheWAS results.

        Generates forest plot showing effect sizes with confidence intervals,
        arranged vertically with statistical information in adjacent panels.
        Automatically detects column names from PheWAS output format. If no
        specific phecodes are provided, automatically selects top positive
        and negative effect values.

        Args:
            phecode_list: Specific phecodes to include in forest plot, auto-selects top effects if None.
            n_top_values: Number of top positive and negative effect values to include when auto-selecting.
            plot_odds_ratio: Whether to plot odds ratio instead of beta for logistic regression results.
            show_phecode: Whether to show phecode alongside phenotype description, defaults to True.
            title: Plot title, auto-generated if None.
            axis_text_size: Font size for axis labels.
            label_size: Font size for text labels.
            marker_shape: Shape of center point markers (default: "s" for square).
            marker_size: Size of center point markers.
            highlight_significance: Whether to use bold text and thicker lines for significant results.
            highlight_phecodes: Specific phecodes to highlight with bold text and thick lines, overrides significance-based highlighting.
            highlight_p_value_threshold: P-value threshold for significance highlighting, defaults to Bonferroni correction if None.
            show_p_value_asterisks: Whether to show significance asterisks next to p-values (* p<0.05, ** p<0.01, *** p<0.001).
            show_count: Whether to show cases/controls panel with N(cases,controls) title.
            show_sex_restriction: Whether to show sex restriction values from phecode data.
            dpi: Plot resolution in dots per inch.
            save_plot: Whether to save plot to file.
            output_file_path: Full path including extension, auto-generated if None.
        """
        get_plot_backend("forest").render(self._ctx, **{k: v for k, v in locals().items() if k != "self"})

    def miami(
            self,
            label_values: str | list[str] = "p_value",
            label_value_threshold: float = 0,
            label_count: int = 10,
            label_size: int = 8,
            label_text_column: str = "phecode_string",
            label_color: str = "label_color",
            label_weight: str = "normal",
            label_split_threshold: int = 30,
            label_box_alpha: float = 0.5,
            marker_alpha: float = 0.7,
            positive_marker_alpha: float | None = None,
            negative_marker_alpha: float | None = None,
            hide_non_significant: bool = False,
            marker_min_size: float = MARKER_MIN_SIZE,
            marker_max_size: float = MARKER_MAX_SIZE,
            effect_cap: float = 10,
            capped_marker_style: str = "circle",
            phecode_categories: list[str] | str | None = None,
            plot_all_categories: bool = True,
            sort_by_significance: bool = False,
            title: str | None = None,
            title_text_size: int = 10,
            y_limit: float | None = None,
            axis_text_size: int = 8,
            show_legend: bool = True,
            show_size_legend: bool = True,
            size_legend_count: int = 4,
            legend_marker_size: int = 7,
            dpi: int = 150,
            save_plot: bool = True,
            output_file_path: str | None = None,
    ) -> None:
        """
        Create Miami plot (mirrored Manhattan) visualization of PheWAS results.

        Generates a mirrored Manhattan plot where positive-effect phecodes plot
        upward and negative-effect phecodes plot downward, with circle markers
        sized by effect magnitude (OR or HR).

        Args:
            label_values: Criteria for labeling points - specific phecodes, "positive_beta", "negative_beta", or "p_value".
            label_value_threshold: Threshold for filtering labels by effect size or p-value.
            label_count: Maximum number of points to label.
            label_size: Font size for data point labels.
            label_text_column: Column containing text for labels.
            label_color: Color specification or column name for label colors.
            label_weight: Font weight for labels.
            label_split_threshold: Character threshold for splitting long labels.
            label_box_alpha: Alpha (transparency) value for label background boxes.
            marker_alpha: Alpha (transparency) value for circle markers (used for both halves).
            positive_marker_alpha: Alpha override for positive effect markers only (upper half). Falls back to marker_alpha if None.
            negative_marker_alpha: Alpha override for negative effect markers only (lower half). Falls back to marker_alpha if None.
            hide_non_significant: If True, omit points below nominal significance (p > 0.05).
            marker_min_size: Minimum marker size for effect magnitude scaling.
            marker_max_size: Maximum marker size for effect magnitude scaling.
            effect_cap: Cap for effect magnitude (OR/HR). Values above this get max dot size.
            capped_marker_style: Marker style for capped values — "circle" (default) renders all points
                as circles with "+" suffix on last size legend entry, "diamond" renders capped points as
                diamonds with a separate legend entry.
            phecode_categories: Specific categories to plot, uses all if None.
            plot_all_categories: Whether to include all categories in plot.
            sort_by_significance: If True, sort phecodes by significance within each category.
            title: Plot title text.
            title_text_size: Font size for plot title.
            y_limit: Maximum absolute y-axis value for display (symmetric).
            axis_text_size: Font size for axis labels.
            show_legend: Whether to display plot legend.
            show_size_legend: Whether to display effect size reference circles in legend.
            size_legend_count: Number of reference circles in size legend.
            legend_marker_size: Size of markers in legend.
            dpi: Plot resolution in dots per inch.
            save_plot: Whether to save plot to file.
            output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), auto-generated if None.
        """
        get_plot_backend("miami").render(self._ctx, **{k: v for k, v in locals().items() if k != "self"})
