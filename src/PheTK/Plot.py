from datetime import datetime
from matplotlib.lines import Line2D
import adjustText
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class Plot:
    def __init__(self,
                 phewas_result_csv_path,
                 bonferroni=None,
                 phecode_version=None,
                 color_palette=None):
        """
        :param phewas_result_csv_path: path to PheWAS result csv file, generated from PheWAS module.
        :param bonferroni: defaults to None; if None, calculate base on number of phecode tested
        :param phecode_version: defaults to None; if None, use phecode X; else phecode 1.2
        :param color_palette: defaults to None; if None, use internal color palette
        """

        # load PheWAS results
        self.phewas_result = pl.read_csv(phewas_result_csv_path, dtypes={"phecode": str})

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

        # bonferroni
        if not bonferroni:
            self.bonferroni = -np.log10(0.05 / len(self.phewas_result))
        else:
            self.bonferroni = bonferroni

        # nominal significance
        self.nominal_significance = -np.log10(0.05)

        # phecode_version
        if phecode_version:
            self.phecode_version = phecode_version.upper()
        else:
            self.phecode_version = "X"

        # phecode categories
        self.phecode_categories = None

        # color mapping
        if color_palette:
            self.color_palette = color_palette
        else:
            self.color_palette = ("blue", "indianred", "darkcyan", "goldenrod", "darkblue", "magenta",
                                  "green", "red", "darkturquoise", "olive", "black", "royalblue",
                                  "maroon", "darkolivegreen", "coral", "purple", "gray")
        self.phecode_categories = self.phewas_result["phecode_category"].unique().to_list()
        self.phecode_categories.sort()
        self.color_dict = {self.phecode_categories[i]: self.color_palette[i % len(self.color_palette)]
                           for i in range(len(self.phecode_categories))}
        self.phewas_result = self.phewas_result.with_columns(
            pl.col("phecode_category").map_dict(self.color_dict).alias("label_color")
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

    @staticmethod
    def save_plot(plot_type="plot", output_file_name=None, output_file_type="pdf"):
        if output_file_name is not None:
            if "." not in output_file_name:
                output_file_name = output_file_name + "." + output_file_type
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{plot_type}_{timestamp}.{output_file_type}"
        plt.savefig(output_file_name, bbox_inches="tight")
        print()
        print("Plot saved to", output_file_name)
        print()

    @staticmethod
    def _filter_by_phecode_categories(df, phecode_categories=None):
        """
        :param df: PheWAS result to filter
        :param phecode_categories: defaults to None, i.e., use all, otherwise, filter what specified
        :return: filtered df
        """
        if phecode_categories:
            if isinstance(phecode_categories, str):
                phecode_categories = [phecode_categories]
            df = df.filter(pl.col("phecode_category").is_in(phecode_categories))
        else:
            df = df

        return df

    @staticmethod
    def _create_phecode_index(df):
        """
        Create phecode index after grouping by phecode_category and phecode;
        Phecode index will be used for plotting purpose
        :param df: PheWAS result to create index
        :return: same dataframe with column "phecode_index" created
        """
        if "phecode_index" in df.columns:
            df = df.drop("phecode_index")
        df = df.sort(by=["phecode_category", "phecode"])\
               .with_columns(pl.Series("phecode_index", range(1, len(df) + 1)))\
               .with_columns(15*np.exp(pl.col("beta")).alias("marker_size_by_beta"))

        return df

    @staticmethod
    def _split_by_beta(df, marker_size_by_beta=False):
        """
        :param df: data of interest, e.g., full phewas result or result of a phecode_category
        :return: positive and negative beta polars dataframes
        """

        # add marker size if marker_size_by_beta is True
        if marker_size_by_beta:
            df = df.with_columns((18*pl.col("beta").abs()).alias("_marker_size"))

        # split to positive and negative beta data
        positive_betas = df.filter(pl.col("beta") >= 0).sort(by="beta", descending=True)
        negative_betas = df.filter(pl.col("beta") < 0).sort(by="beta", descending=False)
        return positive_betas, negative_betas

    @staticmethod
    def _x_ticks(plot_df, selected_color_dict, size=8):
        """
        Generate x tick labels and colors
        :param plot_df: plot data
        :param selected_color_dict: color dict; this is changed based on number of phecode categories selected
        :return: x tick labels and colors for the plot
        """
        x_ticks = plot_df[["phecode_category", "phecode_index"]].groupby("phecode_category").mean()
        # create x ticks labels and colors
        adjustText.plt.xticks(x_ticks["phecode_index"],
                              x_ticks["phecode_category"],
                              rotation=45,
                              ha="right",
                              weight="normal",
                              size=size)
        tick_labels = adjustText.plt.gca().get_xticklabels()
        sorted_labels = sorted(tick_labels, key=lambda label: label.get_text())
        for tick_label, tick_color in zip(sorted_labels, selected_color_dict.values()):
            tick_label.set_color(tick_color)

    def _manhattan_scatter(self, ax, marker_size_by_beta):
        """
        Generate scatter data points
        :param ax: plot object
        :param marker_size_by_beta: adjust marker size by beta coefficient if True
        :return: scatter plot of selected data
        """

        if marker_size_by_beta:
            s_positive = self.positive_betas["_marker_size"]
            s_negative = self.negative_betas["_marker_size"]
        else:
            s_positive = None
            s_negative = None

        ax.scatter(x=self.positive_betas["phecode_index"].to_numpy(),
                   y=self.positive_betas["neg_log_p_value"],
                   s=s_positive,
                   c=self.positive_betas["label_color"],
                   marker="^",
                   alpha=self.positive_alpha)

        ax.scatter(x=self.negative_betas["phecode_index"].to_numpy(),
                   y=self.negative_betas["neg_log_p_value"],
                   s=s_negative,
                   c=self.negative_betas["label_color"],
                   marker="v",
                   alpha=self.negative_alpha)

    def _lines(self,
               ax,
               plot_type,
               plot_df,
               x_col,
               nominal_significance_line=False,
               bonferroni_line=False,
               infinity_line=False,
               y_threshold_line=False,
               y_threshold_value=None,
               x_positive_threshold_line=False,
               x_positive_threshold_value=None,
               x_negative_threshold_line=False,
               x_negative_threshold_value=None):

        extra_offset = 0
        if plot_type == "manhattan":
            extra_offset = 1
        elif plot_type == "volcano":
            extra_offset = 0.05

        # nominal significance line
        if nominal_significance_line:
            ax.hlines(y=-adjustText.np.log10(.05),
                      xmin=plot_df[x_col].min() - self.offset - extra_offset,
                      xmax=plot_df[x_col].max() + self.offset + extra_offset,
                      colors="red",
                      lw=1)

        # bonferroni
        if bonferroni_line:
            ax.hlines(y=self.bonferroni,
                      xmin=plot_df[x_col].min() - self.offset - extra_offset,
                      xmax=plot_df[x_col].max() + self.offset + extra_offset,
                      colors="green",
                      lw=1)

        # infinity
        if infinity_line:
            if self.inf_proxy is not None:
                ax.yaxis.get_major_ticks()[-2].set_visible(False)
                ax.hlines(y=self.inf_proxy * 0.98,
                          xmin=plot_df[x_col].min() - self.offset - extra_offset,
                          xmax=plot_df[x_col].max() + self.offset + extra_offset,
                          colors="blue",
                          linestyle="dashdot",
                          lw=1)

        # y threshold line
        if y_threshold_line:
            ax.hlines(y=y_threshold_value,
                      xmin=plot_df[x_col].min() - self.offset - extra_offset,
                      xmax=plot_df[x_col].max() + self.offset + extra_offset,
                      colors="gray",
                      linestyles="dashed",
                      lw=1)

        # vertical lines
        if x_positive_threshold_line:
            ax.vlines(x=x_positive_threshold_value,
                      ymin=plot_df["neg_log_p_value"].min()-self.offset,
                      ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
                      colors="orange",
                      linestyles="dashed",
                      lw=1)
        if x_negative_threshold_line:
            ax.vlines(x=x_negative_threshold_value,
                      ymin=plot_df["neg_log_p_value"].min()-self.offset,
                      ymax=plot_df["neg_log_p_value"].max() + self.offset + 5,
                      colors="lightseagreen",
                      linestyles="dashed",
                      lw=1)

    @staticmethod
    def _split_text(s, threshold=30):
        """
        Split long text label
        :param s: text string
        :param threshold: approximate number of characters per line
        :return: split text if longer than 40 chars
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

    def _manhattan_label(self,
                         plot_df,
                         label_values,
                         label_count,
                         label_categories=None,
                         label_text_column="phecode_string",
                         label_value_threshold=0,
                         label_split_threshold=30,
                         label_color="label_color",
                         label_size=8,
                         label_weight="normal",
                         y_col="neg_log_p_value",
                         x_col="phecode_index"):
        """
        :param plot_df: plot data
        :param label_values: can take a single phecode, a list of phecodes,
                             or preset values "positive_betas", "negative_betas", "p_value"
        :param label_value_threshold: cutoff value for label values;
                                      if label_values is "positive_beta", keep beta values >= cutoff
                                      if label_values is "negative_beta", keep beta values <= cutoff
                                      if label_values is "p_value", keep neg_log_p_value >= cutoff
        :param label_text_column: defaults to "phecode_string"; name of column contain text for labels
        :param label_count: number of items to label, only needed if label_by input is data type
        :param label_split_threshold: number of characters to consider splitting long labels
        :param label_color: string type; takes either a color or name of column contains color for plot data
        :param label_size: defaults to 8
        :param label_weight: takes standard plt weight inputs, e.g., "normal", "bold", etc.
        :param x_col: column contains x values
        :param y_col: column contains y values
        :return: adjustText object
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
                        positive_betas.filter(pl.col("beta") >= label_value_threshold)
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
                        negative_betas.filter(pl.col("beta") <= label_value_threshold)
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
                self.data_to_label = pl.concat([self.data_to_label,
                                                plot_df.filter(pl.col("phecode") == item)])

        texts = []
        for i in range(len(self.data_to_label)):
            if mc.is_color_like(label_color):
                color = pl.Series(values=[label_color]*len(self.data_to_label))
            else:
                # noinspection PyTypeChecker
                color = self.data_to_label[label_color]
            # noinspection PyTypeChecker
            texts.append(adjustText.plt.text(float(self.data_to_label[x_col][i]),
                                             float(self.data_to_label[y_col][i]),
                                             self._split_text(self.data_to_label[label_text_column][i],
                                                              label_split_threshold),
                                             color=color[i],
                                             size=label_size,
                                             weight=label_weight,
                                             alpha=1,
                                             bbox=dict(facecolor="white",
                                                       edgecolor="none",
                                                       boxstyle="round",
                                                       alpha=0.5,
                                                       lw=0.5)))

        if len(texts) > 0:
            return adjustText.adjust_text(texts,
                                          arrowprops=dict(arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2))

    def _manhattan_legend(self, ax, legend_marker_size):
        """
        :param ax: plot object
        :param legend_marker_size: size of markers
        :return: legend element
        """
        legend_elements = [
            Line2D([0], [0], color="blue", lw=1, linestyle="dashdot", label="Infinity"),
            Line2D([0], [0], color="green", lw=1, label="Bonferroni\nCorrection"),
            Line2D([0], [0], color="red", lw=1, label="Nominal\nSignificance"),
            Line2D([0], [0], marker="^", label="Increased\nRisk Effect", color="white",
                   markerfacecolor="blue", alpha=self.positive_alpha, markersize=legend_marker_size),
            Line2D([0], [0], marker="v", label="Decreased\nRisk Effect", color="white",
                   markerfacecolor="blue", alpha=self.negative_alpha, markersize=legend_marker_size),
        ]
        ax.legend(handles=legend_elements,
                  handlelength=2,
                  loc="center left",
                  bbox_to_anchor=(1, 0.5),
                  fontsize=legend_marker_size)

    def manhattan(self,
                  label_values="p_value",
                  label_value_threshold=0,
                  label_count=10,
                  label_size=8,
                  label_text_column="phecode_string",
                  label_color="label_color",
                  label_weight="normal",
                  label_split_threshold=30,
                  marker_size_by_beta=False,
                  phecode_categories=None,
                  plot_all_categories=True,
                  title=None,
                  title_text_size=10,
                  y_limit=None,
                  axis_text_size=8,
                  show_legend=True,
                  legend_marker_size=6,
                  dpi=150,
                  save_plot=True,
                  output_file_name=None,
                  output_file_type="pdf"):

        ############
        # SETTINGS #
        ############

        # setup some variables based on plot_all_categories and phecode_categories

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
            plot_df = self._create_phecode_index(self.phewas_result)
        else:
            if phecode_categories:
                selected_color_dict = {k: self.color_dict[k] for k in phecode_categories}
                n_categories = len(phecode_categories)
                dpi = None
                # create plot_df containing only necessary data for plotting
                plot_df = self._create_phecode_index(
                    self._filter_by_phecode_categories(
                        self.phewas_result, phecode_categories=phecode_categories
                    )
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

        # set limit for display on y axes
        if y_limit is not None:
            ax.set_ylim(-0.2, y_limit)

        # y axis label
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        # generate positive & negative betas
        self.positive_betas, self.negative_betas = self._split_by_beta(plot_df, marker_size_by_beta)

        ############
        # PLOTTING #
        ############

        # x-axis offset
        adjustText.plt.xlim(float(plot_df["phecode_index"].min()) - self.offset - 1,
                            float(plot_df["phecode_index"].max()) + self.offset + 1)

        # create x ticks labels and colors
        self._x_ticks(plot_df, selected_color_dict)

        # scatter
        self._manhattan_scatter(ax=ax, marker_size_by_beta=marker_size_by_beta)

        # lines
        self._lines(ax=ax,
                    plot_type="manhattan",
                    plot_df=plot_df,
                    x_col="phecode_index",
                    nominal_significance_line=True,
                    bonferroni_line=True,
                    infinity_line=True)

        # labeling
        self._manhattan_label(plot_df=plot_df, label_values=label_values, label_count=label_count,
                              label_text_column=label_text_column, label_categories=label_categories,
                              label_value_threshold=label_value_threshold, label_split_threshold=label_split_threshold,
                              label_size=label_size, label_color=label_color, label_weight=label_weight)

        # legend
        if show_legend:
            self._manhattan_legend(ax, legend_marker_size)

        # save plot
        if save_plot:
            self.save_plot(plot_type="manhattan",
                           output_file_name=output_file_name,
                           output_file_type=output_file_type)

    @staticmethod
    def transform_values(df, col, new_col, new_min, new_max):
        df = df.with_columns(((pl.col(col) - pl.col(col).min())
                              * (new_max - new_min)
                              / (pl.col(col).max() - pl.col(col).min())
                              + new_min).alias(new_col))
        return df

    def _volcano_scatter(self,
                         ax,
                         x_col="log10_odds_ratio",
                         y_col="neg_log_p_value",
                         marker_size_col="cases",
                         marker_shape=".",
                         positive_beta_color="indianred",
                         negative_beta_color="darkcyan",
                         fill_marker=True,
                         marker_alpha=0.5,
                         legend_marker_scale=0.5,
                         legend_label_count=5,
                         show_legend=False):

        # set marker edge and face colors
        if fill_marker:
            positive_face_color = positive_beta_color
            negative_face_color = negative_beta_color
        else:
            positive_face_color = "none"
            negative_face_color = "none"

        # color values for every points
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

    def _volcano_label(self,
                       plot_df,
                       phecode_list=None,
                       phecode_string_list=None,
                       x_col="log10_odds_ratio",
                       y_col="neg_log_p_value",
                       label_count=10,
                       label_text_column="phecode_string",
                       label_split_threshold=30,
                       label_size=8,
                       label_weight="normal",
                       y_threshold=5,
                       x_positive_threshold=None,
                       x_negative_threshold=None):

        # get data for labeling, either use a list of phecodes/phecode names of choice, or use x & y thresholds
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
                                                       alpha=0.5,
                                                       lw=0.5),
                                             alpha=1))
        if len(texts) > 0:
            return adjustText.adjust_text(
                texts, arrowprops=dict(arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2)
            )

    def volcano(self,
                phecode_list=None,
                phecode_string_list=None,
                label_count=10,
                x_col="log10_odds_ratio",
                y_col="neg_log_p_value",
                x_axis_label=None,
                exclude_infinity=False,
                y_threshold=None,
                x_negative_threshold=None,
                x_positive_threshold=None,
                bonferroni_line=False,
                nominal_significance_line=False,
                infinity_line=False,
                y_limit=None,
                title=None,
                title_text_size=None,
                axis_text_size=None,
                marker_size_col="cases",
                marker_shape=".",
                fill_marker=True,
                marker_alpha=0.5,
                show_legend=False,
                legend_marker_scale=0.5,
                legend_label_count=5,
                dpi=150,
                save_plot=True,
                output_file_name=None,
                output_file_type="pdf"):

        # set offset
        self.offset = 0.1

        # create plot
        fig, ax = adjustText.plt.subplots(figsize=(12 * self.ratio, 7), dpi=dpi)

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # set limit for display on y axes
        if y_limit is not None:
            ax.set_ylim(-0.2, y_limit)

        # x, y axis label
        if x_col == "log10_odds_ratio":
            x_axis_label = r"$\log_{10}$(OR)"
        elif (x_col != "log10_odds_ratio") and (x_axis_label is None):
            x_axis_label = x_col
        ax.set_xlabel(x_axis_label, size=axis_text_size)
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        # plot_df
        plot_df = self.phewas_result.clone()
        if exclude_infinity:
            plot_df = plot_df.filter(pl.col("neg_log_p_value") != self.inf_proxy)
        # generate positive & negative betas
        self.positive_betas, self.negative_betas = self._split_by_beta(plot_df)

        ############
        # PLOTTING #
        ############

        # scatter
        self._volcano_scatter(ax=ax,
                              x_col=x_col,
                              marker_size_col=marker_size_col,
                              marker_shape=marker_shape,
                              fill_marker=fill_marker,
                              marker_alpha=marker_alpha,
                              legend_marker_scale=legend_marker_scale,
                              legend_label_count=legend_label_count,
                              show_legend=show_legend)

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
        self._lines(ax=ax,
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
                    infinity_line=infinity_line)

        # labels
        self._volcano_label(phecode_list=phecode_list,
                            phecode_string_list=phecode_string_list,
                            plot_df=plot_df,
                            label_count=label_count,
                            x_col=x_col,
                            y_col=y_col,
                            y_threshold=y_threshold,
                            x_positive_threshold=x_positive_threshold,
                            x_negative_threshold=x_negative_threshold)

        # save plot
        if save_plot:
            self.save_plot(plot_type="volcano",
                           output_file_name=output_file_name,
                           output_file_type=output_file_type)
