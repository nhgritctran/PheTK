from matplotlib.lines import Line2D
import adjustText
import matplotlib.colors as mc
import numpy as np
import polars as pl


class Manhattan:
    def __init__(self,
                 phewas_result,
                 bonferroni=None,
                 phecode_version=None,
                 color_palette=None):
        """
        :param phewas_result: phewas result data; this will be converted to polars dataframe if not already is
        :param bonferroni: defaults to None; if None, calculate base on number of phecode tested
        :param phecode_version: defaults to None; if None, use phecode X; else phecode 1.2
        :param color_palette: defaults to None; if None, use internal color palette
        """

        # sort and add index column for phecode order
        self.phewas_result = self._to_polars(phewas_result)

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

    @staticmethod
    def _to_polars(df):
        """
        Check and convert pandas dataframe object to polars dataframe, if applicable
        :param df: dataframe object
        :return: polars dataframe
        """
        if not isinstance(df, pl.DataFrame):
            return pl.from_pandas(df)
        else:
            return df

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
               .with_columns(pl.Series("phecode_index", range(1, len(df) + 1)))

        return df

    @staticmethod
    def _split_by_beta(df):
        """
        :param df: data of interest, e.g., full phewas result or result of a phecode_category
        :return: positive and negative beta polars dataframes
        """
        # split to positive and negative beta data
        positive_betas = df.filter(pl.col("beta_ind") >= 0).sort(by="beta_ind", descending=True)
        negative_betas = df.filter(pl.col("beta_ind") < 0)
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

    def _scatter(self, ax, plot_df):
        """
        Generate scatter data points
        :param ax: plot object
        :param plot_df: dataframe containing data required for plotting
        :return: scatter plot of selected data
        """
        self.positive_betas, self.negative_betas = self._split_by_beta(plot_df)

        ax.scatter(self.positive_betas["phecode_index"].to_numpy(),
                   self.positive_betas["neg_log_p_value"],
                   c=self.positive_betas["label_color"],
                   marker="^",
                   alpha=self.positive_alpha)
        ax.scatter(self.negative_betas["phecode_index"].to_numpy(),
                   self.negative_betas["neg_log_p_value"],
                   c=self.negative_betas["label_color"],
                   marker="v",
                   alpha=self.negative_alpha)

    def _lines(self, ax, plot_df):
        """
        Generate bonferroni, nominal significance and infinity lines
        :param ax: plot object
        :param plot_df:
        :return:
        """
        # nominal significance line
        ax.hlines(-adjustText.np.log10(.05),
                  0 - self.offset,
                  plot_df["phecode_index"].max() + self.offset + 1,
                  colors="red",
                  lw=1)

        # bonferroni
        ax.hlines(self.bonferroni,
                  0 - self.offset,
                  plot_df["phecode_index"].max() + self.offset + 1,
                  colors="green",
                  lw=1)

        # infinity
        max_neg_log_p_value = plot_df.filter(
            pl.col("phecode_category").is_in(self.phecode_categories)
        )["neg_log_p_value"].max()
        if self.inf_proxy is not None:
            ax.yaxis.get_major_ticks()[-2].set_visible(False)
            ax.hlines(self.inf_proxy * 0.98,
                      0 - self.offset,
                      plot_df["phecode_index"].max() + self.offset + 1,
                      colors="blue",
                      linestyle="dashdot",
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
            if line_length >= threshold:
                new_s += "\n"
                line_length = 0
            else:
                new_s += " "

        return new_s

    def _label(self,
               label_df,
               label_values,
               label_count,
               label_text_column="phecode_string",
               label_value_threshold=0,
               label_split_threshold=30,
               label_color="label_color",
               label_size=8,
               label_weight="normal",
               y_col="neg_log_p_value",
               x_col="phecode_index"):
        """
        :param label_df: plot data
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

        self.data_to_label = pl.DataFrame(schema=label_df.schema)
        for item in label_values:
            if item == "positive_beta":
                self.data_to_label = pl.concat(
                    [
                        self.data_to_label,
                        self.positive_betas.filter(pl.col("beta_ind") >= label_value_threshold)[:label_count]
                    ]
                )
            elif item == "negative_beta":
                self.data_to_label = pl.concat(
                    [
                        self.data_to_label,
                        self.negative_betas.filter(pl.col("beta_ind") <= label_value_threshold)[:label_count]
                    ]
                )
                if label_value_threshold:
                    self.data_to_label = self.data_to_label.filter(pl.col("beta_ind") <= label_value_threshold)
            elif item == "p_value":
                self.data_to_label = pl.concat(
                    [
                        self.data_to_label,
                        label_df.sort(by="p_value")
                               .filter(pl.col("neg_log_p_value") >= label_value_threshold)[:label_count]
                    ]
                )
            else:
                self.data_to_label = pl.concat([self.data_to_label,
                                                label_df.filter(pl.col("phecode") == item)])

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
                                             alpha=1))

        if len(texts) > 0:
            return adjustText.adjust_text(texts,
                                          arrowprops=dict(arrowstyle="simple", color="gray", lw=0.5, mutation_scale=2))

    def _legend(self, ax, legend_marker_size):
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

    def plot(self,
             label_values="positive_beta",
             label_value_threshold=0,
             label_count=10,
             label_size=8,
             label_text_column="phecode_string",
             label_color="label_color",
             label_weight="normal",
             label_split_threshold=30,
             phecode_categories=None,
             plot_all_categories=True,
             title=None,
             title_text_size=10,
             y_limit=None,
             axis_text_size=8,
             show_legend=True,
             legend_marker_size=6):

        ############
        # SETTINGS #
        ############

        # setup some variables based on plot_all_categories and phecode_categories
        dpi = 150
        if plot_all_categories:
            selected_color_dict = self.color_dict
            n_categories = len(self.phewas_result.columns)
            # create plot_df containing only necessary data for plotting
            plot_df = self._create_phecode_index(self.phewas_result)
            if phecode_categories:
                label_df = self._create_phecode_index(
                    self._filter_by_phecode_categories(
                        self.phewas_result, phecode_categories=phecode_categories
                    )
                )
            else:
                label_df = plot_df.clone()
        else:
            if phecode_categories:
                if isinstance(phecode_categories, str):
                    phecode_categories = [phecode_categories]
                phecode_categories.sort()
                self.phecode_categories = phecode_categories
                selected_color_dict = {k: self.color_dict[k] for k in phecode_categories}
                n_categories = len(phecode_categories)
                dpi = None
                # create plot_df containing only necessary data for plotting
                plot_df = self._create_phecode_index(
                    self._filter_by_phecode_categories(
                        self.phewas_result, phecode_categories=phecode_categories
                    )
                )
                label_df = plot_df.clone()
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
        self.positive_betas, self.negative_betas = self._split_by_beta(plot_df)

        ############
        # PLOTTING #
        ############

        # x-axis offset
        adjustText.plt.xlim(float(plot_df["phecode_index"].min()) - self.offset - 1,
                            float(plot_df["phecode_index"].max()) + self.offset + 1)

        # create x ticks labels and colors
        self._x_ticks(plot_df, selected_color_dict)

        # scatter
        self._scatter(ax, plot_df)

        # lines
        self._lines(ax, plot_df)

        # labeling
        self._label(label_df, label_values=label_values, label_count=label_count, label_text_column=label_text_column,
                    label_value_threshold=label_value_threshold, label_split_threshold=label_split_threshold,
                    label_size=label_size, label_color=label_color, label_weight=label_weight)

        # legend
        if show_legend:
            self._legend(ax, legend_marker_size)
