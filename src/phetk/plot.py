from matplotlib.lines import Line2D
import adjustText
import datetime
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _utils


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
        
        :param phewas_result_file_path: Path to PheWAS result CSV/TSV file generated from PheWAS module.
        :type phewas_result_file_path: str
        :param converged_only: Whether to plot converged results only.
        :type converged_only: bool
        :param bonferroni: Bonferroni correction threshold, calculated based on number of phecodes tested if None.
        :type bonferroni: float | None
        :param phecode_version: Phecode version ("1.2" or "X"), defaults to "X" if None.
        :type phecode_version: str | None
        :param color_palette: Color palette - "default", "colorblind", "rainbow", or custom tuple of colors.
        :type color_palette: tuple[str, ...] | str | None
        """

        # load PheWAS results
        sep = _utils.detect_delimiter(phewas_result_file_path)
        self.phewas_result = pl.read_csv(
            phewas_result_file_path,
            separator=sep,
            schema_overrides={"phecode": str, "converged": bool}
        )

        # bonferroni
        if bonferroni is None:
            self.bonferroni = -np.log10(0.05 / len(self.phewas_result))
        else:
            self.bonferroni = bonferroni

        # remove non-converged phecodes - doing this after bonferroni to avoid bonferroni value shifting
        if ("converged" in self.phewas_result.columns) and converged_only:
            self.phewas_result = self.phewas_result.filter(pl.col("converged") == "true")

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
        if color_palette is None or color_palette == "default":
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
            raise ValueError(f"Invalid color_palette: {color_palette}. Use 'default', 'colorblind', 'rainbow', or a tuple of colors.")

        self.phecode_categories = self.phewas_result["phecode_category"].unique().to_list()
        self.phecode_categories.sort()
        self.color_dict = {self.phecode_categories[i]: self.color_palette[i % len(self.color_palette)]
                           for i in range(len(self.phecode_categories))}
        self.phewas_result = self.phewas_result.with_columns(
            pl.col("phecode_category").replace(self.color_dict).alias("label_color")
        )
        self.positive_betas = None
        self.negative_betas = None
        self.data_to_label = None

        # plot element scaling ratio
        self.ratio = 1

        # color lightness
        self.positive_alpha = 0.7
        self.negative_alpha = 0.3

        # offset
        self.offset = 9

        # volcano plot label data
        self.volcano_label_data = None

        # column name for the datapoint direction, e.g., beta for logistic regression or log_hazard_ratio for cox
        self.direction_col = None
        if "beta" in self.phewas_result.columns:
            self.direction_col = "beta"
        elif "log_hazard_ratio" in self.phewas_result.columns:
            self.direction_col = "log_hazard_ratio"

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
        
        :param plot_type: Type of plot for filename generation when auto-generating.
        :type plot_type: str
        :param output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), 
                                auto-generated with timestamp if None.
        :type output_file_path: str | None
        :return: Saves plot file and prints confirmation message.
        :rtype: None
        """
        if output_file_path is not None:
            # If no extension provided, default to PDF
            if "." not in output_file_path:
                output_file_path = output_file_path + ".pdf"
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_path = f"{plot_type}_{timestamp}.pdf"
        
        plt.savefig(output_file_path, bbox_inches="tight")
        print()
        print("Plot saved to", output_file_path)
        print()

    @staticmethod
    def _filter_by_phecode_categories(
            df: pl.DataFrame, 
            phecode_categories: list[str] | str | None = None
    ) -> pl.DataFrame:
        """
        Filter PheWAS results by specified phecode categories.
        
        Restricts dataset to only include results from specified phecode
        categories, useful for focused analysis or plotting.
        
        :param df: PheWAS result dataframe to filter.
        :type df: pl.DataFrame
        :param phecode_categories: Specific phecode categories to include, uses all if None.
        :type phecode_categories: list[str] | str | None
        :return: Filtered dataframe containing only specified categories.
        :rtype: pl.DataFrame
        """
        if phecode_categories:
            if isinstance(phecode_categories, str):
                phecode_categories = [phecode_categories]
            df = df.filter(pl.col("phecode_category").is_in(phecode_categories))
        else:
            df = df

        return df

    def _create_phecode_index(
            self, 
            df: pl.DataFrame,
            sort_by_significance: bool = False
    ) -> pl.DataFrame:
        """
        Create sequential phecode index for Manhattan plot x-axis positioning.
        
        Sorts phecodes by category and assigns sequential indices for proper
        positioning in Manhattan plots, also adds marker size column based on
        effect direction.
        
        :param df: PheWAS result dataframe to create index for.
        :type df: pl.DataFrame
        :param sort_by_significance: If True, sort by phecode_category and -log10(p-value) descending within each category.
        :type sort_by_significance: bool
        :return: Dataframe with phecode_index and marker_size columns added.
        :rtype: pl.DataFrame
        """
        if "phecode_index" in df.columns:
            df = df.drop("phecode_index")
        
        if sort_by_significance:
            df = df.sort(by=["phecode_category", "neg_log_p_value"], descending=[False, True])
        else:
            df = df.sort(by=["phecode_category", "phecode"])
            
        df = df.with_columns(pl.Series("phecode_index", range(1, len(df) + 1)))\
               .with_columns(15*np.exp(pl.col(self.direction_col)).alias(f"marker_size_by_{self.direction_col}"))

        return df

    def _split_by_effect_size(
            self, 
            df: pl.DataFrame, 
            marker_size_by_effect_size: bool = False
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split PheWAS results into positive and negative effect directions.
        
        Separates results based on effect direction (positive/negative beta or
        log hazard ratio) and optionally adds marker size column for visualization.
        
        :param df: PheWAS result dataframe to split.
        :type df: pl.DataFrame
        :param marker_size_by_effect_size: Whether to add marker size column based on effect magnitude.
        :type marker_size_by_effect_size: bool
        :return: Positive effects dataframe and negative effects dataframe.
        :rtype: tuple[pl.DataFrame, pl.DataFrame]
        """

        # add marker size if marker_size_by_effect_size is True
        if marker_size_by_effect_size:
            df = df.with_columns((18*pl.col(self.direction_col).abs()).alias("_marker_size"))

        # split to positive and negative beta data
        positive_betas = df.filter(pl.col(self.direction_col) >= 0).sort(by=self.direction_col, descending=True)
        negative_betas = df.filter(pl.col(self.direction_col) < 0).sort(by=self.direction_col, descending=False)
        return positive_betas, negative_betas

    @staticmethod
    def _x_ticks(
            plot_df: pl.DataFrame, 
            selected_color_dict: dict[str, str], 
            size: int = 8
    ) -> None:
        """
        Generate colored x-axis tick labels for Manhattan plot.
        
        Creates x-axis labels positioned at category centers with colors
        matching phecode category color scheme.
        
        :param plot_df: Plot dataframe containing phecode categories and indices.
        :type plot_df: pl.DataFrame
        :param selected_color_dict: Color mapping for phecode categories.
        :type selected_color_dict: dict[str, str]
        :param size: Font size for tick labels.
        :type size: int
        :return: Sets x-axis ticks and labels with appropriate colors.
        :rtype: None
        """
        x_ticks = plot_df[["phecode_category", "phecode_index"]].group_by("phecode_category").mean()
        # create x ticks labels and colors
        adjustText.plt.xticks(
                x_ticks["phecode_index"],
                x_ticks["phecode_category"],
                rotation=45,
                ha="right",
                weight="normal",
                size=size
            )
        tick_labels = adjustText.plt.gca().get_xticklabels()
        sorted_labels = sorted(tick_labels, key=lambda label: label.get_text())
        for tick_label, tick_color in zip(sorted_labels, selected_color_dict.values()):
            tick_label.set_color(tick_color)

    def _manhattan_scatter(
            self, 
            ax, 
            marker_size_by_effect_size: bool, 
            scale_factor: float = 1
    ) -> None:
        """
        Generate scatter plot points for Manhattan plot.
        
        Creates scatter plot with upward triangles for positive effects and
        downward triangles for negative effects, with optional marker sizing
        based on effect magnitude.
        
        :param ax: Matplotlib axes object for plotting.
        :param marker_size_by_effect_size: Whether to scale marker size by effect magnitude.
        :type marker_size_by_effect_size: bool
        :param scale_factor: Scaling factor for marker sizes.
        :type scale_factor: float
        :return: Adds scatter plot elements to axes.
        :rtype: None
        """

        if marker_size_by_effect_size:
            s_positive = self.positive_betas["_marker_size"] * scale_factor
            s_negative = self.negative_betas["_marker_size"] * scale_factor
        else:
            s_positive = None
            s_negative = None

        ax.scatter(
            x=self.positive_betas["phecode_index"].to_numpy(),
            y=self.positive_betas["neg_log_p_value"],
            s=s_positive,
            c=self.positive_betas["label_color"],
            marker="^",
            alpha=self.positive_alpha
        )

        ax.scatter(
            x=self.negative_betas["phecode_index"].to_numpy(),
            y=self.negative_betas["neg_log_p_value"],
            s=s_negative,
            c=self.negative_betas["label_color"],
            marker="v",
            alpha=self.negative_alpha
        )

    def _lines(
            self,
            ax,
            plot_type: str,
            plot_df: pl.DataFrame,
            x_col: str,
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
        
        Adds horizontal and vertical reference lines including nominal significance,
        Bonferroni correction, infinity proxy, and custom thresholds.
        
        :param ax: Matplotlib axes object for plotting.
        :param plot_type: Type of plot ("manhattan" or "volcano") for offset calculation.
        :type plot_type: str
        :param plot_df: Plot dataframe for determining line extent.
        :type plot_df: pl.DataFrame
        :param x_col: Column name for x-axis values.
        :type x_col: str
        :param nominal_significance_line: Whether to draw nominal significance line (p=0.05).
        :type nominal_significance_line: bool
        :param bonferroni_line: Whether to draw Bonferroni correction line.
        :type bonferroni_line: bool
        :param infinity_line: Whether to draw infinity proxy line.
        :type infinity_line: bool
        :param y_threshold_line: Whether to draw custom y-threshold line.
        :type y_threshold_line: bool
        :param y_threshold_value: Y-value for custom threshold line.
        :type y_threshold_value: float | None
        :param x_positive_threshold_line: Whether to draw positive x-threshold line.
        :type x_positive_threshold_line: bool
        :param x_positive_threshold_value: X-value for positive threshold line.
        :type x_positive_threshold_value: float | None
        :param x_negative_threshold_line: Whether to draw negative x-threshold line.
        :type x_negative_threshold_line: bool
        :param x_negative_threshold_value: X-value for negative threshold line.
        :type x_negative_threshold_value: float | None
        :return: Adds line elements to axes.
        :rtype: None
        """

        extra_offset = 0
        if plot_type == "manhattan":
            extra_offset = 1
        elif plot_type == "volcano":
            extra_offset = 0.05

        # nominal significance line
        if nominal_significance_line:
            ax.hlines(
                y=-adjustText.np.log10(.05),
                xmin=plot_df[x_col].min() - self.offset - extra_offset,
                xmax=plot_df[x_col].max() + self.offset + extra_offset,
                colors="red",
                lw=1
            )

        # bonferroni
        if bonferroni_line:
            ax.hlines(
                y=self.bonferroni,
                xmin=plot_df[x_col].min() - self.offset - extra_offset,
                xmax=plot_df[x_col].max() + self.offset + extra_offset,
                colors="green",
                lw=1
            )

        # infinity
        if infinity_line:
            if self.inf_proxy is not None:
                ax.yaxis.get_major_ticks()[-2].set_visible(False)
                ax.hlines(
                    y=self.inf_proxy * 0.98,
                    xmin=plot_df[x_col].min() - self.offset - extra_offset,
                    xmax=plot_df[x_col].max() + self.offset + extra_offset,
                    colors="blue",
                    linestyle="dashdot",
                    lw=1
                )

        # y threshold line
        if y_threshold_line:
            ax.hlines(
                y=y_threshold_value,
                xmin=plot_df[x_col].min() - self.offset - extra_offset,
                xmax=plot_df[x_col].max() + self.offset + extra_offset,
                colors="gray",
                linestyles="dashed",
                lw=1
            )

        # vertical lines
        if x_positive_threshold_line:
            ax.vlines(
                x=x_positive_threshold_value,
                ymin=plot_df["neg_log_p_value"].min()-self.offset,
                ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
                colors="orange",
                linestyles="dashed",
                lw=1
            )
        if x_negative_threshold_line:
            ax.vlines(
                x=x_negative_threshold_value,
                ymin=plot_df["neg_log_p_value"].min()-self.offset,
                ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
                colors="lightseagreen",
                linestyles="dashed",
                lw=1
            )

    @staticmethod
    def _split_text(
            s: str, 
            threshold: int = 30
    ) -> str:
        """
        Split long text labels into multiple lines for better readability.
        
        Breaks text at word boundaries when line length exceeds threshold,
        improving label appearance in plots.
        
        :param s: Text string to split.
        :type s: str
        :param threshold: Approximate number of characters per line.
        :type threshold: int
        :return: Text with line breaks inserted at appropriate positions.
        :rtype: str
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

    def _manhattan_label(
        self,
        plot_df: pl.DataFrame,
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
        
        Creates text labels for selected data points based on various criteria
        (specific phecodes, effect thresholds, or p-values) with automatic
        positioning to avoid overlaps.
        
        :param plot_df: Plot dataframe containing results to label.
        :type plot_df: pl.DataFrame
        :param label_values: Labeling criteria - specific phecodes, "positive_beta", "negative_beta", or "p_value".
        :type label_values: str | list[str]
        :param label_count: Maximum number of items to label.
        :type label_count: int
        :param label_categories: Specific phecode categories to restrict labeling to.
        :type label_categories: list[str] | None
        :param label_text_column: Column containing text for labels.
        :type label_text_column: str
        :param label_value_threshold: Threshold value for filtering labels by effect size or p-value.
        :type label_value_threshold: float
        :param label_split_threshold: Character count threshold for splitting long labels.
        :type label_split_threshold: int
        :param label_color: Color specification or column name containing colors.
        :type label_color: str
        :param label_size: Font size for labels.
        :type label_size: int
        :param label_weight: Font weight for labels.
        :type label_weight: str
        :param y_col: Column containing y-axis values.
        :type y_col: str
        :param x_col: Column containing x-axis values.
        :type x_col: str
        :return: adjustText object for label positioning.
        """

        if isinstance(label_values, str):
            label_values = [label_values]

        self.data_to_label = pl.DataFrame(schema=plot_df.schema)
        positive_betas = self.positive_betas.clone()
        negative_betas = self.negative_betas.clone()
        if "_marker_size" in positive_betas.columns:
            positive_betas = positive_betas.drop("_marker_size")
        if "_marker_size" in negative_betas.columns:
            negative_betas = negative_betas.drop("_marker_size")

        for item in label_values:
            if item == "positive_beta":
                self.data_to_label = pl.concat(
                        [
                            self.data_to_label,
                            positive_betas.filter(pl.col(self.direction_col) >= label_value_threshold)
                        ]
                    )
                if label_categories is not None:
                    self.data_to_label = self.data_to_label.filter(
                        pl.col("phecode_category").is_in(label_categories)
                    )[:label_count]
                else:
                    self.data_to_label = self.data_to_label[:label_count]
            elif item == "negative_beta":
                self.data_to_label = pl.concat(
                        [
                            self.data_to_label,
                            negative_betas.filter(pl.col(self.direction_col) <= label_value_threshold)
                        ]
                    )
                if label_categories is not None:
                    self.data_to_label = self.data_to_label.filter(
                        pl.col("phecode_category").is_in(label_categories)
                    )[:label_count]
                else:
                    self.data_to_label = self.data_to_label[:label_count]
            elif item == "p_value":
                self.data_to_label = pl.concat(
                    [
                        self.data_to_label,
                        plot_df.sort(by="p_value")
                               .filter(pl.col("neg_log_p_value") >= label_value_threshold)
                    ]
                )
                if label_categories is not None:
                    self.data_to_label = self.data_to_label.filter(
                        pl.col("phecode_category").is_in(label_categories)
                    )[:label_count]
                else:
                    self.data_to_label = self.data_to_label[:label_count]
            else:
                self.data_to_label = pl.concat(
                        [self.data_to_label,
                         plot_df.filter(pl.col("phecode") == item)]
                    )

        texts = []
        for i in range(len(self.data_to_label)):
            if mc.is_color_like(label_color):
                color = pl.Series(values=[label_color]*len(self.data_to_label))
            else:
                # noinspection PyTypeChecker
                color = self.data_to_label[label_color]
            # noinspection PyTypeChecker
            texts.append(
                adjustText.plt.text(
                    float(self.data_to_label[x_col][i]),
                    float(self.data_to_label[y_col][i]),
                    self._split_text(
                        self.data_to_label[label_text_column][i],
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
            self, 
            ax, 
            legend_marker_size: int
    ) -> None:
        """
        Create legend for Manhattan plot with significance lines and effect markers.
        
        Adds legend explaining infinity line, Bonferroni correction, nominal
        significance, and effect direction markers.
        
        :param ax: Matplotlib axes object for legend.
        :param legend_marker_size: Size of markers in legend.
        :type legend_marker_size: int
        :return: Adds legend to plot.
        :rtype: None
        """
        legend_elements = []
        
        # Only add infinity line to legend if infinity values are present
        if self.inf_proxy is not None:
            legend_elements.append(Line2D([0], [0], color="blue", lw=1, linestyle="dashdot", label="Infinity"))
        
        legend_elements.extend([
            Line2D([0], [0], color="green", lw=1, label="Bonferroni\nCorrection"),
            Line2D([0], [0], color="red", lw=1, label="Nominal\nSignificance"),
            Line2D([0], [0], marker="^", label="Increased\nRisk Effect", color="white",
                   markerfacecolor="blue", alpha=self.positive_alpha, markersize=legend_marker_size),
            Line2D([0], [0], marker="v", label="Decreased\nRisk Effect", color="white",
                   markerfacecolor="blue", alpha=self.negative_alpha, markersize=legend_marker_size),
        ])
        ax.legend(
            handles=legend_elements,
            handlelength=2,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=legend_marker_size
        )

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
            phecode_categories: list[str] | str | None = None,
            plot_all_categories: bool = True,
            sort_by_significance: bool = False,
            title: str | None = None,
            title_text_size: int = 10,
            y_limit: float | None = None,
            axis_text_size: int = 8,
            show_legend: bool = True,
            legend_marker_size: int = 6,
            dpi: int = 150,
            save_plot: bool = True,
            output_file_path: str | None = None
    ) -> None:
        """
        Create Manhattan plot visualization of PheWAS results.
        
        Generates comprehensive Manhattan plot showing -log10(p-values) across
        phecode categories with customizable labeling, significance lines,
        and effect direction indicators.
        
        :param label_values: Criteria for labeling points - specific phecodes, "positive_beta", "negative_beta", or "p_value".
        :type label_values: str | list[str]
        :param label_value_threshold: Threshold for filtering labels by effect size or p-value.
        :type label_value_threshold: float
        :param label_count: Maximum number of points to label.
        :type label_count: int
        :param label_size: Font size for data point labels.
        :type label_size: int
        :param label_text_column: Column containing text for labels.
        :type label_text_column: str
        :param label_color: Color specification or column name for label colors.
        :type label_color: str
        :param label_weight: Font weight for labels.
        :type label_weight: str
        :param label_split_threshold: Character threshold for splitting long labels.
        :type label_split_threshold: int
        :param label_box_alpha: Alpha (transparency) value for label background boxes.
        :type label_box_alpha: float
        :param marker_size_by_effect_size: Whether to scale marker size by effect magnitude.
        :type marker_size_by_effect_size: bool
        :param marker_scale_factor: Scaling factor for marker sizes.
        :type marker_scale_factor: float
        :param positive_marker_alpha: Alpha (transparency) value for positive effect markers.
        :type positive_marker_alpha: float
        :param negative_marker_alpha: Alpha (transparency) value for negative effect markers.
        :type negative_marker_alpha: float
        :param phecode_categories: Specific categories to plot, uses all if None.
        :type phecode_categories: list[str] | str | None
        :param plot_all_categories: Whether to include all categories in plot.
        :type plot_all_categories: bool
        :param sort_by_significance: If True, sort phecodes by significance within each category.
        :type sort_by_significance: bool
        :param title: Plot title text.
        :type title: str | None
        :param title_text_size: Font size for plot title.
        :type title_text_size: int
        :param y_limit: Maximum y-axis value for display.
        :type y_limit: float | None
        :param axis_text_size: Font size for axis labels.
        :type axis_text_size: int
        :param show_legend: Whether to display plot legend.
        :type show_legend: bool
        :param legend_marker_size: Size of markers in legend.
        :type legend_marker_size: int
        :param dpi: Plot resolution in dots per inch.
        :type dpi: int
        :param save_plot: Whether to save plot to file.
        :type save_plot: bool
        :param output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), auto-generated if None.
        :type output_file_path: str | None
        :return: Creates and optionally saves Manhattan plot.
        :rtype: None
        """

        ############
        # SETTINGS #
        ############

        # Set up some variables based on plot_all_categories and phecode_categories

        # offset
        self.offset = 9
        
        # phecode_categories & label_categories
        if phecode_categories:
            if isinstance(phecode_categories, str):  # convert to list if input is str
                phecode_categories = [phecode_categories]
            phecode_categories.sort()
            label_categories = phecode_categories
            self.phecode_categories = phecode_categories
        else:
            label_categories = None
        
        # plot_df and label_value_cols
        if plot_all_categories:
            selected_color_dict = self.color_dict
            n_categories = len(self.phewas_result.columns)
            # create plot_df containing only necessary data for plotting
            plot_df = self._create_phecode_index(self.phewas_result, sort_by_significance)
        else:
            if phecode_categories:
                selected_color_dict = {k: self.color_dict[k] for k in phecode_categories}
                n_categories = len(phecode_categories)
                dpi = None
                # create plot_df containing only necessary data for plotting
                plot_df = self._create_phecode_index(
                    self._filter_by_phecode_categories(
                        self.phewas_result, phecode_categories=phecode_categories
                    ), sort_by_significance
                )
            else:
                print("phecode_categories must not be None when plot_all_categories = False.")
                return

        # create plot
        self.ratio = (n_categories/len(self.phewas_result.columns))
        fig, ax = adjustText.plt.subplots(figsize=(12*self.ratio, 7), dpi=dpi)

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
        self.positive_betas, self.negative_betas = self._split_by_effect_size(plot_df, marker_size_by_effect_size)
        
        # Update alpha values from parameters
        self.positive_alpha = positive_marker_alpha
        self.negative_alpha = negative_marker_alpha

        ############
        # PLOTTING #
        ############

        # x-axis offset
        adjustText.plt.xlim(float(plot_df["phecode_index"].min()) - self.offset - 1,
                            float(plot_df["phecode_index"].max()) + self.offset + 1)

        # create x ticks labels and colors
        self._x_ticks(plot_df, selected_color_dict)

        # scatter
        self._manhattan_scatter(ax=ax, marker_size_by_effect_size=marker_size_by_effect_size, scale_factor=marker_scale_factor)

        # lines
        self._lines(
            ax=ax,
            plot_type="manhattan",
            plot_df=plot_df,
            x_col="phecode_index",
            nominal_significance_line=True,
            bonferroni_line=True,
            infinity_line=True
        )

        # labeling
        self._manhattan_label(
            plot_df=plot_df,
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
            self._manhattan_legend(ax, legend_marker_size)

        # save plot
        if save_plot:
            self.save_plot(plot_type="manhattan", output_file_path=output_file_path)

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
        
        :param df: Input dataframe containing column to transform.
        :type df: pl.DataFrame
        :param col: Name of column to transform.
        :type col: str
        :param new_col: Name for new column containing transformed values.
        :type new_col: str
        :param new_min: Minimum value for transformed range.
        :type new_min: float
        :param new_max: Maximum value for transformed range.
        :type new_max: float
        :return: Dataframe with additional column containing transformed values.
        :rtype: pl.DataFrame
        """
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).min())
                 * (new_max - new_min)
                 / (pl.col(col).max() - pl.col(col).min())
                 + new_min
            ).alias(new_col))
        return df

    def _volcano_scatter(
            self,
            ax,
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
        
        Generates scatter plot with points colored by effect direction and
        optionally sized by case count or other variables.
        
        :param ax: Matplotlib axes object for plotting.
        :param x_col: Column name for x-axis values (effect size).
        :type x_col: str
        :param y_col: Column name for y-axis values (significance).
        :type y_col: str
        :param marker_size_col: Column name for marker sizing, constant size if None.
        :type marker_size_col: str | None
        :param marker_shape: Shape of markers for plotting.
        :type marker_shape: str
        :param positive_beta_color: Color for positive effect markers.
        :type positive_beta_color: str
        :param negative_beta_color: Color for negative effect markers.
        :type negative_beta_color: str
        :param fill_marker: Whether to fill markers with color.
        :type fill_marker: bool
        :param marker_alpha: Transparency level for markers.
        :type marker_alpha: float
        :param legend_marker_scale: Scale factor for legend markers.
        :type legend_marker_scale: float
        :param legend_label_count: Number of items in size legend.
        :type legend_label_count: int
        :param show_legend: Whether to display size legend.
        :type show_legend: bool
        :return: Adds scatter plot elements to axes.
        :rtype: None
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
        pos_df = self.positive_betas[col_list].with_columns(pl.lit(positive_beta_color)
                                                            .alias("edge_color"))\
                                              .with_columns(pl.lit(positive_face_color)
                                                            .alias("face_color"))
        neg_df = self.negative_betas[col_list].with_columns(pl.lit(negative_beta_color)
                                                            .alias("edge_color"))\
                                              .with_columns(pl.lit(negative_face_color)
                                                            .alias("face_color"))
        # combined into 1 df for plotting
        full_df = pl.concat([pos_df, neg_df]).unique()
        if marker_size_col is not None:
            # scale values for better visualization
            full_df = self.transform_values(df=full_df,
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
        step_size = (max_size - min_size) * (1-(2*k)) / (legend_label_count - 1)
        legend_labels = [
            min_size + margin + (i * step_size) for i in range(legend_label_count)
        ]
        legend_labels = [round(i, -2) for i in legend_labels]
        if (marker_size_col is not None) and show_legend:
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, num=legend_label_count)
            ax.legend(
                handles=handles,
                labels=legend_labels,  # override with original values
                markerscale=legend_marker_scale,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title=marker_size_col
            )

    def _volcano_label(
            self,
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
            y_threshold: float = 5,
            x_positive_threshold: float | None = None,
            x_negative_threshold: float | None = None
    ):
        """
        Add data point labels to volcano plot with automatic positioning.
        
        Creates text labels for data points based on specific phecode lists
        or threshold criteria, with color coding by effect direction.
        
        :param plot_df: Plot dataframe containing results to label.
        :type plot_df: pl.DataFrame
        :param phecode_list: Specific phecodes to label.
        :type phecode_list: list[str] | str | None
        :param phecode_string_list: Specific phecode descriptions to label.
        :type phecode_string_list: list[str] | str | None
        :param x_col: Column name for x-axis values.
        :type x_col: str
        :param y_col: Column name for y-axis values.
        :type y_col: str
        :param label_count: Maximum number of labels to display.
        :type label_count: int
        :param label_text_column: Column containing text for labels.
        :type label_text_column: str
        :param label_split_threshold: Character threshold for splitting long labels.
        :type label_split_threshold: int
        :param label_size: Font size for labels.
        :type label_size: int
        :param label_weight: Font weight for labels.
        :type label_weight: str
        :param y_threshold: Minimum significance threshold for labeling.
        :type y_threshold: float
        :param x_positive_threshold: Minimum positive effect threshold for labeling.
        :type x_positive_threshold: float | None
        :param x_negative_threshold: Maximum negative effect threshold for labeling.
        :type x_negative_threshold: float | None
        :return: adjustText object for label positioning.
        """

        # Get the data for labeling, either use a list of phecodes/phecode names of choice or use x & y thresholds
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
                [data_to_label.top_k(by=x_col, descending=True, k=round(label_count/2), nulls_last=True),
                 data_to_label.top_k(by=x_col, descending=False, k=round(label_count/2), nulls_last=True)]
            ).unique()
        else:
            data_to_label = pl.concat(
                [plot_df.top_k(by=x_col, descending=True, k=round(label_count/2), nulls_last=True),
                 plot_df.top_k(by=x_col, descending=False, k=round(label_count/2), nulls_last=True)]
            ).unique()

        # label data
        self.volcano_label_data = data_to_label

        texts = []
        for i in range(len(data_to_label)):
            if data_to_label[x_col][i] < 0:
                color = "green"
            else:
                color = "red"
            # noinspection PyTypeChecker
            texts.append(adjustText.plt.text(float(data_to_label[x_col][i]),
                                             float(data_to_label[y_col][i]),
                                             self._split_text(data_to_label[label_text_column][i],
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
        
        :param phecode_list: Specific phecodes to label on plot.
        :type phecode_list: list[str] | str | None
        :param phecode_string_list: Specific phecode descriptions to label.
        :type phecode_string_list: list[str] | str | None
        :param label_count: Maximum number of points to label.
        :type label_count: int
        :param x_col: Column name for x-axis values (effect size).
        :type x_col: str
        :param y_col: Column name for y-axis values (significance).
        :type y_col: str
        :param x_axis_label: Custom label for x-axis.
        :type x_axis_label: str | None
        :param exclude_infinity: Whether to exclude infinite significance values.
        :type exclude_infinity: bool
        :param y_threshold: Significance threshold for labeling and reference line.
        :type y_threshold: float | None
        :param x_negative_threshold: Negative effect threshold for labeling and reference line.
        :type x_negative_threshold: float | None
        :param x_positive_threshold: Positive effect threshold for labeling and reference line.
        :type x_positive_threshold: float | None
        :param bonferroni_line: Whether to display Bonferroni correction line.
        :type bonferroni_line: bool
        :param nominal_significance_line: Whether to display nominal significance line.
        :type nominal_significance_line: bool
        :param infinity_line: Whether to display infinity proxy line.
        :type infinity_line: bool
        :param y_limit: Maximum y-axis value for display.
        :type y_limit: float | None
        :param title: Plot title text.
        :type title: str | None
        :param title_text_size: Font size for plot title.
        :type title_text_size: int | None
        :param axis_text_size: Font size for axis labels.
        :type axis_text_size: int | None
        :param marker_size_col: Column name for marker sizing.
        :type marker_size_col: str | None
        :param marker_shape: Shape of markers for plotting.
        :type marker_shape: str
        :param fill_marker: Whether to fill markers with color.
        :type fill_marker: bool
        :param marker_alpha: Transparency level for markers.
        :type marker_alpha: float
        :param label_box_alpha: Alpha (transparency) value for label background boxes.
        :type label_box_alpha: float
        :param show_legend: Whether to display size legend.
        :type show_legend: bool
        :param legend_marker_scale: Scale factor for legend markers.
        :type legend_marker_scale: float
        :param legend_label_count: Number of items in size legend.
        :type legend_label_count: int
        :param dpi: Plot resolution in dots per inch.
        :type dpi: int
        :param save_plot: Whether to save plot to file.
        :type save_plot: bool
        :param output_file_path: Full path including extension (e.g., "plot.png", "results.pdf"), auto-generated if None.
        :type output_file_path: str | None
        :return: Creates and optionally saves volcano plot.
        :rtype: None
        """

        # set offset
        self.offset = 0.1

        # create plot
        fig, ax = adjustText.plt.subplots(figsize=(12 * self.ratio, 7), dpi=dpi)

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # plot_df
        plot_df = self.phewas_result.clone()
        if exclude_infinity:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") != self.inf_proxy)
        
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
        self.positive_betas, self.negative_betas = self._split_by_effect_size(plot_df)

        ############
        # PLOTTING #
        ############

        # scatter
        self._volcano_scatter(
            ax=ax,
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
        x_positive_threshold_line = False
        x_negative_threshold_line = False
        y_threshold_line = False
        if x_positive_threshold is not None:
            x_positive_threshold_line = True
        if x_negative_threshold is not None:
            x_negative_threshold_line = True
        if y_threshold is not None:
            y_threshold_line = True
        self._lines(
            ax=ax,
            plot_type="volcano",
            plot_df=plot_df,
            x_col=x_col,
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
        self._volcano_label(
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
        if save_plot:
            self.save_plot(plot_type="volcano", output_file_path=output_file_path)

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
        
        :param phecode_list: Specific phecodes to include in forest plot, auto-selects top effects if None.
        :type phecode_list: list[str] | str | None
        :param n_top_values: Number of top positive and negative effect values to include when auto-selecting.
        :type n_top_values: int
        :param plot_odds_ratio: Whether to plot odds ratio instead of beta for logistic regression results.
        :type plot_odds_ratio: bool
        :param show_phecode: Whether to show phecode alongside phenotype description, defaults to True.
        :type show_phecode: bool
        :param title: Plot title, auto-generated if None.
        :type title: str | None
        :param axis_text_size: Font size for axis labels.
        :type axis_text_size: int
        :param label_size: Font size for text labels.
        :type label_size: int
        :param marker_shape: Shape of center point markers (default: "s" for square).
        :type marker_shape: str
        :param marker_size: Size of center point markers.
        :type marker_size: int
        :param highlight_significance: Whether to use bold text and thicker lines for significant results.
        :type highlight_significance: bool
        :param highlight_phecodes: Specific phecodes to highlight with bold text and thick lines, overrides significance-based highlighting.
        :type highlight_phecodes: list[str] | str | None
        :param highlight_p_value_threshold: P-value threshold for significance highlighting, defaults to Bonferroni correction if None.
        :type highlight_p_value_threshold: float | None
        :param show_p_value_asterisks: Whether to show significance asterisks next to p-values (* p<0.05, ** p<0.01, *** p<0.001).
        :type show_p_value_asterisks: bool
        :param show_count: Whether to show cases/controls panel with N(cases,controls) title.
        :type show_count: bool
        :param show_sex_restriction: Whether to show sex restriction values from phecode data.
        :type show_sex_restriction: bool
        :param dpi: Plot resolution in dots per inch.
        :type dpi: int
        :param save_plot: Whether to save plot to file.
        :type save_plot: bool
        :param output_file_path: Full path including extension, auto-generated if None.
        :type output_file_path: str | None
        :return: Creates and optionally saves forest plot.
        :rtype: None
        """
        
        
        
        # Auto-select top positive and negative effects if no phecode_list provided
        if phecode_list is None:
            # First detect the effect column to determine direction
            if "beta" in self.phewas_result.columns:
                effect_col = "beta"
            elif "hazard_ratio" in self.phewas_result.columns:
                effect_col = "hazard_ratio"
            elif "log_hazard_ratio" in self.phewas_result.columns:
                effect_col = "log_hazard_ratio"
            else:
                print("Could not find effect size column (beta, hazard_ratio, or log_hazard_ratio)")
                return
            
            # Get top positive and negative phecode lists
            if effect_col == "hazard_ratio":
                positive_phecodes = self.phewas_result.filter(pl.col(effect_col) > 1).top_k(
                    by=effect_col, k=n_top_values, reverse=False
                )["phecode"].to_list()
                negative_phecodes = self.phewas_result.filter(pl.col(effect_col) < 1).top_k(
                    by=effect_col, k=n_top_values, reverse=True
                )["phecode"].to_list()
            else:
                positive_phecodes = self.phewas_result.filter(pl.col(effect_col) > 0).top_k(
                    by=effect_col, k=n_top_values, reverse=False
                )["phecode"].to_list()
                negative_phecodes = self.phewas_result.filter(pl.col(effect_col) < 0).top_k(
                    by=effect_col, k=n_top_values, reverse=True
                )["phecode"].to_list()
            
            # Combine phecode lists
            phecode_list = positive_phecodes + negative_phecodes
            
            if len(phecode_list) == 0:
                print("No data found with positive or negative effects.")
                return
        else:
            # Convert single phecode to list
            if isinstance(phecode_list, str):
                phecode_list = [phecode_list]
        
        # Filter data for specified phecodes
        plot_data = self.phewas_result.filter(pl.col("phecode").is_in(phecode_list))
        
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
                    # Calculate odds ratio from beta
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
            # Logistic regression format
            if effect_col == "odds_ratio":
                # Calculate odds ratio confidence intervals from beta CIs
                plot_data = plot_data.with_columns([
                    pl.col("conf_int_1").exp().alias("odds_ratio_ci_low"),
                    pl.col("conf_int_2").exp().alias("odds_ratio_ci_high")
                ])
                ci_cols = ("odds_ratio_ci_low", "odds_ratio_ci_high")
            else:
                ci_cols = ("conf_int_1", "conf_int_2")
        elif "hazard_ratio_low" in plot_data.columns and "hazard_ratio_high" in plot_data.columns:
            # Cox regression format
            ci_cols = ("hazard_ratio_low", "hazard_ratio_high")
        elif "standard_error" in plot_data.columns:
            # Calculate from beta and standard error if available
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
        
        # Extract data for plotting - automatically show appropriate effect measure
        effects = plot_data[effect_col].to_numpy()
        if effect_col == "hazard_ratio" or effect_col == "odds_ratio":
            reference_line = 1.0
        else:
            reference_line = 0.0
        x_label = effect_type
        
        ci_lows = plot_data[ci_cols[0]].to_numpy()
        ci_highs = plot_data[ci_cols[1]].to_numpy()
        p_values = plot_data["p_value"].to_numpy()
        # Format phenotype text based on show_phecode parameter
        phecodes = plot_data["phecode"].to_list()
        if show_phecode:
            phecode_strings = [f"{string} ({phecode})" for string, phecode in zip(plot_data["phecode_string"].to_list(), phecodes)]
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
            highlight_p_value_threshold = 10 ** (-self.bonferroni)
        
        # Create figure with subplots
        n_phecodes = len(plot_data)
        
        # Determine number of panels and width ratios
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
        
        # Note: We no longer use log scale for hazard ratios
        
        # Plot confidence intervals as horizontal lines
        for i, (effect, ci_low, ci_high, pval, phecode) in enumerate(zip(effects, ci_lows, ci_highs, p_values, phecodes)):
            # Determine line thickness and marker edge width based on highlighting conditions
            should_highlight = (highlight_significance and pval <= highlight_p_value_threshold) or (phecode in highlight_phecodes_set)
            if should_highlight:
                line_width = 2
                marker_edge_width = 1.5
            else:
                line_width = 1
                marker_edge_width = 1
            
            # Confidence interval line
            ax_forest.plot([ci_low, ci_high], [i, i], 'k-', linewidth=line_width, alpha=0.7)

            # Point estimate - color based on neutral value
            color = Plot._get_marker_color(effect, effect_col)
            
            ax_forest.plot(effect, i, marker_shape, color=color, markersize=marker_size, 
                          markeredgecolor='black', linewidth=marker_edge_width)
            
            # Add caps to confidence interval
            cap_height = 0.1
            ax_forest.plot([ci_low, ci_low], [i-cap_height, i+cap_height], 'k-', linewidth=line_width)
            ax_forest.plot([ci_high, ci_high], [i-cap_height, i+cap_height], 'k-', linewidth=line_width)
        
        # Add reference line
        ax_forest.axvline(x=reference_line, color='black', linestyle='--', alpha=0.5)
        
        # Format forest plot
        ax_forest.set_xlabel(x_label, fontweight='bold', fontsize=axis_text_size)
        ax_forest.set_yticks([])
        ax_forest.set_ylim(-0.5, n_phecodes - 0.5)
        ax_forest.grid(False)
        ax_forest.invert_yaxis()
        
        # Remove top and right spines
        ax_forest.spines['top'].set_visible(False)
        ax_forest.spines['right'].set_visible(False)
        ax_forest.spines['left'].set_visible(False)
        
        # Note: No special formatting needed since we're not using log scale
        
        # Text panel (first panel) - Phecode descriptions
        panel_title = 'Phenotype (phecode)   ' if show_phecode else 'Phenotype   '
        ax_text.set_title(panel_title, fontweight='bold', fontsize=axis_text_size, loc="right")
        Plot._setup_panel_and_add_text(
            ax_text, '', phecode_strings, p_values, phecodes, effects,
            highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
            effect_col, n_phecodes, axis_text_size, label_size, ha='right'
        )
        
        # Effect (CI) panel (third panel)
        effect_texts = []
        for effect, ci_low, ci_high in zip(effects, ci_lows, ci_highs):
            effect_text = f"{effect:.3f} ({ci_low:.3f}, {ci_high:.3f})"
            effect_texts.append(effect_text)
        
        Plot._setup_panel_and_add_text(ax_effect, f'{effect_type} (95% CI)', effect_texts, p_values, phecodes, effects,
                                 highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                 effect_col, n_phecodes, axis_text_size, label_size)
        
        # p-value panel (fourth panel)
        pval_texts = []
        for pval in p_values:
            pval_text = f"{pval:.2e}"
            if show_p_value_asterisks:
                asterisks = Plot._get_p_value_asterisks(pval)
                pval_text += asterisks
            pval_texts.append(pval_text)
        
        Plot._setup_panel_and_add_text(ax_pval, 'p-value', pval_texts, p_values, phecodes, effects,
                                 highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                 effect_col, n_phecodes, axis_text_size, label_size)
        
        # Cases/Controls panel (fifth panel if show_count is True)
        if show_count:
            count_texts = []
            for cases, controls in zip(plot_data["cases"], plot_data["controls"]):
                total_n = cases + controls
                count_text = f"{total_n:,}({cases:,}/{controls:,})"
                count_texts.append(count_text)
            
            Plot._setup_panel_and_add_text(ax_count, 'N(cases/controls)', count_texts, p_values, phecodes, effects,
                                     highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                     effect_col, n_phecodes, axis_text_size, label_size)
        
        # Sex restriction panel (last panel if show_sex_restriction is True)
        if show_sex_restriction:
            Plot._setup_panel_and_add_text(ax_sex, 'Sex', plot_data["phecode_sex_restriction"], p_values, phecodes, effects,
                                     highlight_significance, highlight_p_value_threshold, highlight_phecodes_set,
                                     effect_col, n_phecodes, axis_text_size, label_size)

        # Save plot
        if save_plot:
            self.save_plot(plot_type="forest", output_file_path=output_file_path)
    
    @staticmethod
    def _get_marker_color(effect_val: float, effect_column: str, positive_color: str = '#D55E00', negative_color: str = '#56B4E9') -> str:
        """Determine marker color based on effect size and regression type."""
        if effect_column == "hazard_ratio" or effect_column == "odds_ratio":
            return positive_color if effect_val > 1.0 else negative_color
        else:
            return positive_color if effect_val > 0.0 else negative_color
    
    @staticmethod
    def _get_text_color(effect_val: float, effect_column: str, highlight: bool) -> str:
        """Determine text color based on highlighting and effect direction."""
        if highlight:
            return Plot._get_marker_color(effect_val, effect_column)
        else:
            return 'black'
    
    @staticmethod
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
    
    @staticmethod
    def _setup_panel_and_add_text(
            ax, title: str, texts: list[str], p_values, phecodes, effects,
            highlight_significance: bool, highlight_p_value_threshold: float,
            highlight_phecodes_set: set, effect_col: str, n_phecodes: int,
            axis_text_size: int, label_size: int, ha: str = 'center'
    ):
        """Setup panel formatting and add text with highlighting."""
        ax.set_title(title, fontweight='bold', fontsize=axis_text_size)
        
        for i, (text, pval, phecode, effect) in enumerate(zip(texts, p_values, phecodes, effects)):
            should_highlight = (highlight_significance and pval <= highlight_p_value_threshold) or (phecode in highlight_phecodes_set)
            weight = 'bold' if should_highlight else 'normal'
            text_color = Plot._get_text_color(effect, effect_col, should_highlight)
            
            ax.text(0.5 if ha == 'center' else 0.95, i, text, va='center', ha=ha, 
                   fontsize=label_size, weight=weight, color=text_color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_phecodes - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.invert_yaxis()
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
