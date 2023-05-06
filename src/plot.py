from matplotlib.lines import Line2D
import adjustText
import colorsys
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
            pl.col("phecode_category").map_dict(self.color_dict).alias("color")
        )

        self.ratio = 1

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

    def _split_by_beta(self, df):
        """
        :param df: data of interest, e.g., full phewas result or result of a phecode_category
        :return: positive and negative beta polars dataframes
        """
        # split to positive and negative beta data
        self.positive_betas = df.filter(pl.col("beta_ind") >= 0)
        self.negative_betas = df.filter(pl.col("beta_ind") < 0)
        return self.positive_betas, self.negative_betas

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
        :param plot_df: dataframe containing data required for plotting
        :return: scatter plot of selected data
        """
        self.positive_betas, self.negative_betas = self._split_by_beta(plot_df)

        ax.scatter(self.positive_betas["phecode_index"].to_numpy(),
                   self.positive_betas["neg_log_p_value"],
                   c=self.positive_betas["color"],
                   marker="^",
                   alpha=.3)
        ax.scatter(self.negative_betas["phecode_index"].to_numpy(),
                   self.negative_betas["neg_log_p_value"],
                   c=self.negative_betas["color"],
                   marker="v",
                   alpha=.3)

    @staticmethod
    def _adjust_lightness(color, amount=0.5):
        """
        adjust color lightness
        :param color: str, color name, in "color" column in result table
        :param amount: defaults to 0.5
        :return: color from HSV coordinates to RGB coordinates
        """
        try:
            c = mc.cnames[color]
        except NameError:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    @staticmethod
    def _split_long_text(s):
        """
        split long text label
        :param s: text string
        :return: split text if longer than 40 chars
        """
        if len(s) > 40:
            words = s.split(" ")
            mid = len(words) // 2
            first_half = " ".join(words[:mid])
            second_half = " ".join(words[mid:])
            full_text = first_half + "\n" + second_half
            return full_text
        else:
            return s

    def _label_data(self,
                    xcol,
                    ycol,
                    dcol,
                    ccol,
                    label_size=10,
                    label_weight="normal"):
        """
        method to label data
        ---
        input
            df: dataframe contains data points for labeling
            xcol: column contains x values
            ycol: column contains y values
            dcol: column contains values for labeling
            label_size: label size
            label_weight: label weight
        ---
        output
            labels in plot via adjust_text function
        """

        texts = []
        for i in range(len(self.phewas_result)):
            # set value for color variable
            if mc.is_color_like(ccol):
                color = ccol
            else:
                color = self.phewas_result[ccol].iloc[i]

            # create texts variable
            texts.append(adjustText.plt.text(float(self.phewas_result[xcol].iloc[i]),
                                             self.phewas_result[ycol].iloc[i],
                                             self._split_long_text(self.phewas_result[dcol].iloc[i]),
                                             color=color,
                                             size=label_size,
                                             weight=label_weight))

        return adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    def plot(self,
             phecode_categories=None,
             title=None,
             show_legend=True,
             y_limit=None,
             axis_text_size=8,
             legend_marker_size=6,
             title_text_size=10):

        #################
        # MISC SETTINGS #
        #################

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

        ############
        # PLOTTING #
        ############

        # create x ticks labels and colors
        self._x_ticks(plot_df, selected_color_dict)

        # scatter
        self._scatter(ax, plot_df)

        ##########
        # LEGEND #
        ##########
        if show_legend:
            if not phecode_categories:
                legend_elements = [Line2D([0], [0], color="b", lw=2, linestyle="dashdot", label="Infinity"),
                                   Line2D([0], [0], color="g", lw=2, label="Bonferroni Correction"),
                                   Line2D([0], [0], color="r", lw=2, label="Nominal Significance Level"),
                                   Line2D([0], [0], marker="^", label="Increased Risk Effect",
                                          color="b", markerfacecolor="b", markersize=legend_marker_size),
                                   Line2D([0], [0], marker="v", label="Decreased Risk Effect",
                                          color="b", markerfacecolor="b", markersize=legend_marker_size), ]
            else:
                legend_elements = [Line2D([0], [0], color="b", lw=2, linestyle="dashdot", label="Infinity"),
                                   Line2D([0], [0], color="g", lw=2, label="Bonferroni Correction"),
                                   Line2D([0], [0], color="r", lw=2, label="Nominal Significance Level"),
                                   Line2D([0], [0], marker="^", label="Increased Risk Effect",
                                          markerfacecolor="b", markersize=legend_marker_size),
                                   Line2D([0], [0], marker="v", label="Decreased Risk Effect",
                                          markerfacecolor="b", markersize=legend_marker_size), ]
            ax.legend(handles=legend_elements, handlelength=2, loc="center left", bbox_to_anchor=(1, 0.5), size=6)
