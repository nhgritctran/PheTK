from IPython.display import display
from matplotlib.lines import Line2D
import adjustText
import colorsys
import matplotlib.colors as mc
import numpy as np
import pandas as pd
import polars as pl


class Manhattan:
    def __init__(self,
                 phewas_result,
                 bonferroni=None,
                 phecode_version=None):
        self.phewas_result = phewas_result
        # sort and add index column for phecode order
        self.phewas_result = self.phewas_result\
            .sort(by=["phecode_category", "phecode"])\
            .with_columns(pl.Series("phecode_index", range(1, len(self.phewas_result) + 1)))

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

        # create plot
        self.fig, self.ax = adjustText.plt.subplots(figsize=(20, 10))

        # y axis label
        self.ax.set_ylabel(r"$-\log_{10}$(p-value)", size=12)

    def _split_by_beta(self, df):
        """
        :param df: data of interest, e.g., full phewas result or result of a phecode_category
        :return: positive and negative beta polars dataframes
        """
        # split to positive and negative beta data
        self.positive_betas = df.filter(pl.col("beta_ind") >= 0)
        self.negative_betas = df.filter(pl.col("beta_ind") < 0)
        return self.negative_betas, self.negative_betas

    def _scatter(self, phecode_category=None):
        """
        generate scatter data points
        :param phecode_category: defaults to None, i.e., use all categories
        :return: scatter plot of selected data
        """
        if phecode_category:
            self.positive_betas, self.negative_betas = self._split_by_beta(
                self.phewas_result.filter(pl.col("phecode_category") == phecode_category)
            )
        else:
            self.positive_betas, self.negative_betas = self._split_by_beta(self.phewas_result)

        self.ax.scatter(self.positive_betas["phecode_index"].to_numpy(),
                        self.positive_betas["neg_log_p_value"],
                        marker="^",
                        alpha=.3)
        self.ax.scatter(self.negative_betas["phecode_index"].to_numpy(),
                        self.negative_betas["neg_log_p_value"],
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

    def _map_color(self):
        color_dict = {"Auditory": "blue",
                      "Cardiovascular": "indianred",
                      "Complications of care": "darkcyan",
                      "Congenital": "goldenrod",
                      "Dermatologic": "darkblue",
                      "Developmental": "magenta",
                      "Endocrine": "green",
                      "Gastrointestinal": "red",
                      "Genitourinary": "darkturquoise",
                      "Haematopoietic": "olive",
                      "Infectious": "black",
                      "Metabolic": "royalblue",
                      "Musculoskeletal": "maroon",
                      "Neonate": "darkolivegreen",
                      "Neoplastic": "coral",
                      "Neurologic": "purple",
                      "Ophthalmologic": "gray",
                      "Pregnancy": "darkcyan",
                      "Psychiatric": "darkorange",
                      "Pulmonary": "coral",
                      "Rx": "chartreuse",
                      "Signs/Symptoms": "firebrick",
                      "Statistics": "mediumspringgreen",
                      "Traumatic": "gray"}
        self.phewas_result["color"] = self.phewas_result["phecode_category"].map(color_dict)
        return self.phewas_result

    def plot(self,
             phecode_category=None,
             title=None,
             show_legend=True,
             y_limit=None):

        #################
        # MISC SETTINGS #
        #################

        # plot title
        if title is not None:
            adjustText.plt.title(title, weight="bold", size=16)

        # set limit for display on y axes
        if y_limit is not None:
            self.ax.set_ylim(-0.2, y_limit)

        ############
        # PLOTTING #
        ############

        self._scatter(phecode_category)

        ##########
        # LEGEND #
        ##########

        if show_legend:
            if not phecode_category:
                legend_elements = [Line2D([0], [0], color="b", lw=2, linestyle="dashdot", label="Infinity"),
                                   Line2D([0], [0], color="g", lw=2, label="Bonferroni Correction"),
                                   Line2D([0], [0], color="r", lw=2, label="Nominal Significance Level"),
                                   Line2D([0], [0], marker="v", label="Decreased Risk Effect",
                                          color="b", markerfacecolor="b", markersize=12),
                                   Line2D([0], [0], marker="^", label="Increased Risk Effect",
                                          color="b", markerfacecolor="b", markersize=12), ]
            else:
                legend_elements = [Line2D([0], [0], color="b", lw=2, linestyle="dashdot", label="Infinity"),
                                   Line2D([0], [0], color="g", lw=2, label="Bonferroni Correction"),
                                   Line2D([0], [0], color="r", lw=2, label="Nominal Significance Level"),
                                   Line2D([0], [0], marker="v", label="Decreased Risk Effect",
                                          markerfacecolor="b", markersize=12),
                                   Line2D([0], [0], marker="^", label="Increased Risk Effect",
                                          markerfacecolor="b", markersize=12), ]
            self.ax.legend(handles=legend_elements, handlelength=2, loc="center left", bbox_to_anchor=(1, 0.5))
