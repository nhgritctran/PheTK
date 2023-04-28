from IPython.display import display
from matplotlib.lines import Line2D
import adjustText
import colorsys
import matplotlib.colors as mc
import pandas as pd


def adjust_lightness(color, amount=0.5):
    """
    adjust color lightness
    :param color: str, color name, in 'color' column in result table
    :param amount: defaults to 0.5
    :return: color from HSV coordinates to RGB coordinates
    """
    try:
        c = mc.cnames[color]
    except NameError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def return_table(df, phenotypes, name="", as_or=False):
    """
    method to manipulate input dataframe
    :param df: input dataframe
    :param phenotypes: series, phenotype phecode_string
    :param name: str
    :param as_or: defaults to False
    :return: table with new columns, including "β" or "OR"
    """
    '''
    method to manipulate input dataframe
    ---
    input
        df: dataframe
        phenotypes: series, phenotype phecode_string
        name: str
    ---
    output
        return table with new columns, including "β" or "OR"
    '''
    if not as_or:
        # print(phenotypes)
        ret = df.loc[df["phecode_string"].isin(phenotypes)][["index", "phecode", "phecode_string",
                                                          "phecode_category", "cases", "controls",
                                                          "beta_ind", "conf_int_1",
                                                          "conf_int_2", "p_value", "color"]]
        #         display(ret.head())
        ret["β"] = (adjustText.np.round(ret['beta_ind'], decimals=2).apply(str) + " (" +
                    adjustText.np.round(ret['conf_int_1'], decimals=2).apply(str) + ", " +
                    adjustText.np.round(ret['conf_int_2'], decimals=2).apply(str) + ")")

        ret = ret.drop(["beta_ind", "conf_int_1", "conf_int_2"], axis=1)
        ret.columns = [str(col) + "_" + name for col in ret.columns]
    else:
        ret = df.loc[df["phecode_string"].isin(phenotypes)][["index", "phecode", "phecode_string",
                                                          "phecode_category", "cases", "controls",
                                                          "beta_ind", "conf_int_1",
                                                          "conf_int_2", "p_value", "color"]]
        ret["OR"] = (adjustText.np.round(adjustText.np.exp(ret['beta_ind']), decimals=2).apply(
            str) + " (" +
                     adjustText.np.round(adjustText.np.exp(ret['conf_int_1']), decimals=2).apply(
                         str) + ", " +
                     adjustText.np.round(adjustText.np.exp(ret['conf_int_2']), decimals=2).apply(
                         str) + ")")
        # ret = ret.drop(["beta_ind","conf_int_1","conf_int_2" ], axis=1)
        ret.columns = [str(col) + "_" + name for col in ret.columns]
    return ret


def top_phenotypes(df_this, name="top", num=10, by_beta_abs=True):
    """
    method to get top phenotypes
    ---
    input
        df_this: dataframe
        name: str, table name, default = 'top'
        num: int, number of phenotypes to select, default = 10
        by_beta_abs: boolean, check 'beta_abs' column
    ---
    ouput
        top phenotype table
    """

    # This line might give you empty phenotypes
    # df_this = df_this[df_this["neg_p_log_10"] >= bonf]
    df_this["beta_abs"] = adjustText.np.abs(df_this["beta_ind"])
    if by_beta_abs:
        top = df_this.sort_values(["beta_abs"], ascending=False)
    else:
        top = df_this.sort_values(["p_value"], ascending=True)
    table_formatted = return_table(df=top, phenotypes=top["phecode_string"], name=name)
    return table_formatted.head(num)


def split_long_text(s):
    """
    method to split long text, used for labeling phecode phecode_string
    ---
    input
        s: tring variable, e.g., phecode phecode_string
    ---
    output
        s if len(s) < 40, else splitted into 2 lines at mid point
    """
    if len(s) > 40:
        words = s.split(" ")
        mid = len(words) // 2
        first_half = " ".join(words[:mid])
        second_half = " ".join(words[mid:])
        splitted = first_half + "\n" + second_half
        return (splitted)
    else:
        return s


def label_data(df,
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
    for i in range(len(df)):
        # set value for color variable
        if mc.is_color_like(ccol):
            color = ccol
        else:
            color = df[ccol].iloc[i]

        # create texts variable
        texts.append(adjustText.plt.text(float(df[xcol].iloc[i]),
                                         df[ycol].iloc[i],
                                         split_long_text(df[dcol].iloc[i]),
                                         color=color,
                                         size=label_size,
                                         weight=label_weight))

    return adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))


def map_color(df):
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
                  "Pregnancy": "blue",
                  "Psychiatric": "indianred",
                  "Pulmonary": "darkcyan",
                  "Rx": "goldenrod",
                  "Signs/Symptoms": "darkblue",
                  "Statistics": "magenta",
                  "Traumatic": "green"}
    df["color"] = df["phecode_category"].map(color_dict)
    return df


def manhattan_plot(phewas_result,
                   bonferroni,
                   phecode_category="all",
                   annotate="phecode_string",
                   by_beta_abs=True,  # Otherwise by p_value
                   size_beta=True,
                   show_neg_beta=True,
                   xtick_val="phecode",
                   n_labels=10,
                   label_size=10,
                   label_weight="normal",
                   label_color=None,
                   title=None,
                   show_legend=True,
                   ylim=None):
    """
    method for plotting Manhattan Plot
    ---
    input
        phecode_category: list of phecode_category to display (e.g. all, neoplasms)
        annotate: "phecode_string" - phecode phecode_string
                  "phecode" - phecode
                  a list of phecode - custom list to annotate
    ---
    output
        Manhattan plot
    """

    #################
    # MISC SETTINGS #
    #################

    # for now, turn off this warning
    pd.options.mode.chained_assignment = None  # default='warn'

    # colors
    PheWAS_results_ehr = map_color(phewas_result.copy())

    # sort and add index column for phecode order
    PheWAS_results_ehr = PheWAS_results_ehr.sort_values("code_val").reset_index(drop=True).reset_index()

    # boferroni
    # bonf_corr = .05/phecode_counts["phecode"].nunique() # based on total phecodes available
    # bonf_corr = .05 / PheWAS_results_ehr["phecode"].nunique()  # based on number phecodes in PheWAS result
    bonf_corr = bonferroni

    # initilize plot
    fig, ax = adjustText.plt.subplots(figsize=(20, 10))

    # column to get value for x axis limit
    if xtick_val == "phecode":
        xlim_col_name = "code_val"
    else:
        xlim_col_name = xtick_val

    # threshold lines x offset
    line_x_offset = 9

    # set limit for display on y axes
    if ylim is not None:
        ax.set_ylim(-0.2, ylim)

    # y axis label
    ax.set_ylabel(r"$-\log_{10}$(p-value)", size=12)

    # plot title
    if title is not None:
        adjustText.plt.title(title, weight="bold", size=16)

    ###################
    # DATA PROCESSING #
    ###################

    # subset to particular phecode_category
    if phecode_category != "all":
        PheWAS_results_ehr = PheWAS_results_ehr.loc[PheWAS_results_ehr["phecode_category"] == phecode_category]

    # now separate into positive and negative effect sizes
    pos_beta = PheWAS_results_ehr.loc[PheWAS_results_ehr["beta_ind"] >= 0]
    neg_beta = PheWAS_results_ehr.loc[PheWAS_results_ehr["beta_ind"] < 0]

    # now set colors if only one phecode_category
    if phecode_category != "all":
        pos_color = adjust_lightness(pos_beta["color"].iloc[0], amount=.5)
        neg_color = adjust_lightness(neg_beta["color"].iloc[0], amount=1.)

    # which to annotate
    num = n_labels
    neg_beta_top = top_phenotypes(neg_beta, num=num, by_beta_abs=by_beta_abs)
    pos_beta_top = top_phenotypes(pos_beta, num=num, by_beta_abs=by_beta_abs)

    # check to see if any infs in the p-values
    if sum(adjustText.np.isinf(PheWAS_results_ehr["neg_p_log_10"])) > 0:
        # if the p-values are 0, then map to max non-inf+c for plotting
        inf_map = \
        adjustText.np.sort(adjustText.np.unique(PheWAS_results_ehr["neg_p_log_10"]))[::-1][1] + 50
        inf_map_plot = inf_map + 10
        pos_beta["neg_p_log_10"] = adjustText.np.where(
            adjustText.np.isinf(pos_beta["neg_p_log_10"]),
            inf_map_plot,
            pos_beta["neg_p_log_10"])

        neg_beta["neg_p_log_10"] = adjustText.np.where(
            adjustText.np.isinf(neg_beta["neg_p_log_10"]),
            inf_map_plot,
            neg_beta["neg_p_log_10"])
        pos_beta_top["p_value_top"] = adjustText.np.where(pos_beta_top["p_value_top"] == 0,
                                                                     10 ** (-inf_map_plot),
                                                                     pos_beta_top["p_value_top"])

        neg_beta_top["p_value_top"] = adjustText.np.where(neg_beta_top["p_value_top"] == 0,
                                                                     10 ** (-inf_map_plot),
                                                                     neg_beta_top["p_value_top"])
        # infinity line
        ax.hlines(inf_map,
                  0 - line_x_offset,
                  PheWAS_results_ehr[xlim_col_name].max() + line_x_offset,
                  linestyles="dashdot",
                  colors="b")

    #############
    # TOP BETAS #
    #############

    max_val = PheWAS_results_ehr["neg_p_log_10"].loc[
        PheWAS_results_ehr["neg_p_log_10"] != adjustText.np.inf].max()
    adjustText.np.seterr(divide='ignore')
    neg_beta_top["neg_log"] = -adjustText.np.log10(neg_beta_top["p_value_top"])
    neg_beta_top["neg_log"].loc[neg_beta_top["neg_log"] == adjustText.np.inf] = max_val + 60
    pos_beta_top["neg_log"] = -adjustText.np.log10(pos_beta_top["p_value_top"])
    pos_beta_top["neg_log"].loc[pos_beta_top["neg_log"] == adjustText.np.inf] = max_val + 60
    adjustText.np.seterr(divide='warn')

    print("\033[1m", "Top positive betas:", "\033[0m")
    display(pos_beta_top.head(10).reset_index(drop=True))
    print()
    print("==========")
    print()
    print("\033[1m", "Top negative betas:", "\033[0m")
    display(neg_beta_top.head(10).reset_index(drop=True))
    print()
    print("==========")
    print()
    print("\033[1m", "Manhattan plot:", "\033[0m")

    ############
    # PLOTTING #
    ############

    # need to scale this
    if xtick_val == "phecode":
        plot_val = "code_val"
    else:
        plot_val = xtick_val
    if phecode_category != "all":
        ax.scatter(pos_beta[plot_val],
                   pos_beta["neg_p_log_10"],
                   s=150 * adjustText.np.exp(pos_beta['beta_ind']) if size_beta else 100,
                   c=pos_color, marker='^',
                   alpha=.3)
        ax.scatter(neg_beta[plot_val],
                   neg_beta["neg_p_log_10"],
                   s=150 * adjustText.np.exp(neg_beta['beta_ind']) if size_beta else 100,
                   c=neg_color, marker='v',
                   alpha=.3)
    else:
        ax.scatter(pos_beta[plot_val],
                   pos_beta["neg_p_log_10"],
                   s=15 * adjustText.np.exp(pos_beta['beta_ind']) if size_beta else 100,
                   c=pos_beta['color'],
                   marker='^', alpha=.3)
        ax.scatter(neg_beta[plot_val],
                   neg_beta["neg_p_log_10"],
                   s=15 * adjustText.np.exp(neg_beta['beta_ind']) if size_beta else 100,
                   c=neg_beta['color'],
                   marker='v',
                   alpha=.3)

    # nominal significance line
    ax.hlines(-adjustText.np.log10(.05),
              0 - line_x_offset,
              PheWAS_results_ehr[xlim_col_name].max() + line_x_offset,
              colors="r",
              label="0.05")

    # bonferroni line
    ax.hlines(-adjustText.np.log10(bonf_corr),
              0 - line_x_offset,
              PheWAS_results_ehr[xlim_col_name].max() + line_x_offset,
              colors="g",
              label="Bonferroni Threshold")

    # xticks
    # dataframe for x ticks
    PheWas_ticks = PheWAS_results_ehr[["index", "phecode", "phecode_category"]].astype({"phecode": float})
    # remove certain phecodes to avoid skewing the tick positions
    PheWas_ticks = PheWas_ticks.loc[~PheWas_ticks["phecode"].isin([860,
                                                                   931,
                                                                   938,
                                                                   938.1,
                                                                   938.2,
                                                                   939,
                                                                   939.1,
                                                                   947,
                                                                   980])]
    # group and get mean of phecode for use as tick position
    PheWas_ticks = PheWas_ticks.groupby("phecode_category", as_index=False).mean()
    # reshape the final plot to just fit the phecodes in the subgroup
    adjustText.plt.xlim(float(PheWAS_results_ehr[xtick_val].min()) - line_x_offset - 1,
                        float(PheWAS_results_ehr[xtick_val].max()) + line_x_offset + 1)
    # x axes ticks
    adjustText.plt.xticks(PheWas_ticks[xtick_val],
                          PheWas_ticks["phecode_category"],
                          rotation=45,
                          ha="right",
                          weight="bold",
                          size=12)
    # tick colors
    tickcolors = PheWAS_results_ehr.sort_values("phecode_category")["color"].unique().tolist()
    for ticklabel, tickcolor in zip(adjustText.plt.gca().get_xticklabels(), tickcolors):
        ticklabel.set_color(tickcolor)

    ##############
    # ANNOTATION #
    ##############

    # set label color
    if label_color:
        ccol = label_color
    else:
        ccol = "color_top"

    if annotate == "phecode":
        # set xcol values
        xcol = xtick_val + "_top"

        # label data
        label_data(pos_beta_top,
                   xcol=xcol,
                   ycol="neg_log",
                   dcol="phecode_top",
                   ccol=ccol,
                   label_size=label_size,
                   label_weight=label_weight)
        if show_neg_beta:
            label_data(neg_beta_top,
                       xcol=xcol,
                       ycol="neg_log",
                       dcol="phecode_top",
                       ccol=ccol,
                       label_size=label_size,
                       label_weight=label_weight)

    elif annotate == "phecode_string":
        # set xcol values
        xcol = xtick_val + "_top"

        # label data
        label_data(pos_beta_top,
                   xcol=xcol,
                   ycol="neg_log",
                   dcol="phecode_string_top",
                   ccol=ccol,
                   label_size=label_size,
                   label_weight=label_weight)
        if show_neg_beta:
            label_data(neg_beta_top,
                       xcol=xcol,
                       ycol="neg_log",
                       dcol="phecode_string_top",
                       ccol=ccol,
                       label_size=label_size,
                       label_weight=label_weight)

    else:
        # set x values
        if xtick_val == "phecode":
            xcol = "code_val"
        else:
            xcol = xtick_val

            # pass what's in the list
        res = PheWAS_results_ehr[
            PheWAS_results_ehr["code_val"].isin(annotate)
        ][["index", "code_val", "neg_p_log_10", "phecode_string", "color"]]
        res["color_top"] = res["color"].copy()

        ## drop infs
        res = res[~adjustText.np.isinf(res["neg_p_log_10"])]

        # label data
        label_data(res,
                   xcol=xcol,
                   ycol="neg_p_log_10",
                   dcol="phecode_string",
                   ccol=ccol,
                   label_size=label_size,
                   label_weight=label_weight)

    ##########
    # LEGEND #
    ##########

    if show_legend:
        if phecode_category != "all":
            legend_elements = [Line2D([0], [0], color='b', lw=4,
                                      label='Infinity'),
                               Line2D([0], [0], color='g', lw=4,
                                      label='Bonferroni Correction'),
                               Line2D([0], [0], color='r', lw=4,
                                      label='Nominal Significance Level'),
                               Line2D([0], [0], marker='v',
                                      label='Decreased Risk Effect',
                                      color=neg_color,
                                      markerfacecolor=neg_color, markersize=15),
                               Line2D([0], [0], marker='^',
                                      label='Increased Risk Effect',
                                      color=pos_color,
                                      markerfacecolor=pos_color, markersize=15), ]
        else:
            legend_elements = [Line2D([0], [0], color='b', lw=2, linestyle="dashdot",
                                      label='Infinity'),
                               Line2D([0], [0], color='g', lw=2,
                                      label='Bonferroni Correction'),
                               Line2D([0], [0], color='r', lw=2,
                                      label='Nominal Significance Level'),
                               Line2D([0], [0], marker='v',
                                      label='Decreased Risk Effect',
                                      markerfacecolor='b', markersize=12),
                               Line2D([0], [0], marker='^',
                                      label='Increased Risk Effect',
                                      markerfacecolor='b', markersize=12), ]
        ax.legend(handles=legend_elements, handlelength=2, loc="center left", bbox_to_anchor=(1, 0.5))
