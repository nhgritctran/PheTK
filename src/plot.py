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

        # sort and add index column for phecode order
        self.phewas_result = phewas_result

        # bonferroni
        if not bonferroni:
            self.bonferroni = -np.log10(0.05 / len(self.phewas_result))
        else:
            self.bonferroni = bonferroni

        # phecode_version
        if phecode_version:
            self.phecode_version = phecode_version
        else:
            self.phecode_version = "X"

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
        create phecode index after grouping by phecode_category and phecode;
        phecode index will be used for plotting purpose
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
        generate x tick labels and colors
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
        generate scatter data points
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
        generate bonferroni, nominal significance and infinity lines
        :param ax: plot object
        :param plot_df:
        :return:
        """
        # nominal significance line
        ax.hlines(-adjustText.np.log10(.05),
                  0 - self.offset,
                  plot_df["phecode_index"].max() + self.offset + 1,
                  colors="r")
        # bonferroni
        ax.hlines(self.bonferroni,
                  0 - self.offset,
                  plot_df["phecode_index"].max() + self.offset + 1,
                  colors="g")

    @staticmethod
    def _split_text(s, threshold=40):
        """
        split long text label
        :param s: text string
        :return: split text if longer than 40 chars
        """
        new_s = ""
        element_count = len(s)//threshold+1
        if element_count > 1:
            for i in range(element_count):
                element = s[i*threshold: (i+1)*threshold]
                if i == element_count-1 or element[-1] == " ":
                    new_s += element + "\n"
                else:
                    new_s += element + "-" + "\n"
        else:
            new_s = s

        return new_s

    def _label(self,
               plot_df,
               label_values,
               label_col,
               label_count,
               y_col="neg_log_p_value",
               x_col="phecode_index",
               label_color="label_color",
               label_size=8,
               label_weight="normal"):
        """
        :param plot_df: plot data
        :param label_values: can take a single phecode, a list of phecodes,
                             or preset values "positive_betas", "negative_betas", "p_value"
        :param label_count: number of items to label, only needed if label_by input is data type
        :param x_col: column contains x values
        :param y_col: column contains y values
        :param label_size: defaults to 8
        :param label_weight: takes standard plt weight inputs, e.g., "normal", "bold", etc.
        :return: adjustText object
        """

        if isinstance(label_values, str):
            label_values = [label_values]

        self.data_to_label = pl.DataFrame(schema=plot_df.schema)
        for item in label_values:
            if item == "positive_beta":
                self.data_to_label = pl.concat([self.data_to_label, self.positive_betas[:label_count]])
            elif item == "negative_beta":
                self.data_to_label = pl.concat([self.data_to_label, self.negative_betas[:label_count]])
            elif item == "p_value":
                self.data_to_label = pl.concat([self.data_to_label, plot_df.sort(by="p_value")[:label_count]])
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
                                             self._split_text(self.data_to_label[label_col][i]),
                                             color=color[i],
                                             size=label_size,
                                             weight=label_weight,
                                             alpha=1))

        return adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    def _legend(self, ax, legend_marker_size):
        legend_elements = [Line2D([0], [0], color="b", lw=1, linestyle="dashdot", label="Infinity"),
                           Line2D([0], [0], color="g", lw=1, label="Bonferroni\nCorrection"),
                           Line2D([0], [0], color="r", lw=1, label="Nominal\nSignificance"),
                           Line2D([0], [0], marker="^", label="Increased\nRisk Effect", color="b",
                                  markerfacecolor="b", alpha=self.positive_alpha, markersize=legend_marker_size),
                           Line2D([0], [0], marker="v", label="Decreased\nRisk Effect", color="b",
                                  markerfacecolor="b", alpha=self.negative_alpha, markersize=legend_marker_size), ]
        ax.legend(handles=legend_elements,
                  handlelength=2,
                  loc="center left",
                  bbox_to_anchor=(1, 0.5),
                  fontsize=legend_marker_size)

    def plot(self,
             label_values="positive_beta",
             label_count=10,
             label_column="phecode_string",
             label_color="label_color",
             phecode_categories=None,
             title=None,
             show_legend=True,
             y_limit=None,
             axis_text_size=8,
             legend_marker_size=6,
             title_text_size=10):

        ############
        # SETTINGS #
        ############

        # setup some variables based on phecode_categories
        if phecode_categories:
            if isinstance(phecode_categories, str):
                phecode_categories = [phecode_categories]
            selected_color_dict = {k: self.color_dict[k] for k in phecode_categories}
            n_categories = len(phecode_categories)
        else:
            selected_color_dict = self.color_dict
            n_categories = len(self.phewas_result.columns)

        # create plot
        self.ratio = (n_categories/len(self.phewas_result.columns))
        if phecode_categories:
            dpi = None
        else:
            dpi = 150
        fig, ax = adjustText.plt.subplots(figsize=(12*self.ratio, 7), dpi=dpi)

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=title_text_size)

        # set limit for display on y axes
        if y_limit is not None:
            ax.set_ylim(-0.2, y_limit)

        # y axis label
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        # create plot_df containing only necessary data for plotting
        plot_df = self._create_phecode_index(
            self._filter_by_phecode_categories(
                self.phewas_result, phecode_categories
            )
        )

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
        self._label(plot_df,
                    label_values=label_values,
                    label_count=label_count,
                    label_col=label_column,
                    label_color=label_color)

        # legend
        if show_legend:
            self._legend(ax, legend_marker_size)
