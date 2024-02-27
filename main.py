import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from pyrepo_mcda.mcda_methods import VIKOR
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from daria import DARIA


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 2, 2)
    stacked = True
    width = 0.6
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
        ncol = 2
    else:
        ncol = 5
    
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (10,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results/bar_chart_' + title[-4:] + '.png')
    plt.show()



# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlBu_r",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '.png')
    plt.show()


def main():
    
    path = 'dataset'

    # Symbols of Countries
    coun_names = pd.read_csv('dataset/country_names.csv')
    country_names = list(coun_names['Country'])
    # Number of countries
    m = len(country_names)

    str_years = [str(y) for y in range(2020, 2023)]
    # dataframe for annual results VIKOR
    preferences_v = pd.DataFrame(index = country_names)
    rankings_v = pd.DataFrame(index = country_names)


    # initialization of the VIKOR method object
    vikor = VIKOR()

    # dataframes for results summary
    summary_corrs = pd.DataFrame(index = country_names)

    for el, year in enumerate(str_years):
        file = 'healthcare_dataset_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        # types: 1 denotes profit and -1 denotes cost
        types = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1])
        
        list_of_cols = list(data.columns)
        # decision matrix
        matrix = data.to_numpy()
        # weights calculated by CRITIC method
        weights = mcda_weights.critic_weighting(matrix)

        # VIKOR annual
        pref = vikor(matrix, weights, types)
        rank = rank_preferences(pref, reverse = False)
        
        preferences_v[year] = pref
        rankings_v[year] = rank
        summary_corrs['VIKOR ' + str(year)] = rank

    preferences_v = preferences_v.rename_axis('Country')
    preferences_v.to_csv('results/preferences_v.csv')
    rankings_v = rankings_v.rename_axis('Country')
    rankings_v.to_csv('results/rankings_v.csv')
    
    # PLOT VIKOR results =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 1, 1)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (7, 6))
    for i in range(rankings_v.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_v.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max + 0.2, rankings_v.iloc[i, -1]),
                        fontsize = 12, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, str_years, fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('VIKOR annual rankings')
    plt.tight_layout()
    plt.savefig('results/rankings_years_v' + '.png')
    plt.show()
    
    
    # ======================================================================
    # Temporal VIKOR method
    # ======================================================================
    # DARIA (DAta vaRIAbility) temporal approach
    # preferences includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = country_names)
    df = preferences_v.T
    matrix = df.to_numpy()

    # VIKOR orders preferences in ascending order
    met = 'vikor'
    type = -1

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values with Entropy selecting variability measure
    var = daria._stat_var(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, type)

    # for next stage of research
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_results = pd.DataFrame()
    df_results['Ai'] = list(df.columns)
    df_results['Variability'] = list(var)
    
    # list of directions
    df_results['dir list'] = dir_list
    
    df_results.to_csv('results/scores_v.csv')
    df_varia_fin = df_varia_fin.rename_axis('Country')
    df_varia_fin.to_csv('results/FINAL_V.csv')

    # final calculation
    # data with alternatives' rankings' variability values calculated with Gini coeff and directions
    G_df = copy.deepcopy(df_varia_fin)

    # data with alternatives' efficiency of performance calculated for the recent period
    S_df = copy.deepcopy(preferences_v)

    # ==============================================================
    # S = S_df.mean(axis = 1).to_numpy()
    S = S_df['2022'].to_numpy()

    G = G_df[met.upper()].to_numpy()
    dir = G_df[met.upper() + ' dir'].to_numpy()

    # update efficiencies using DARIA methodology
    # final updated preferences
    final_S = daria._update_efficiency(S, G, dir)

    # VIKOR has ascending ranking from prefs
    rank = rank_preferences(final_S, reverse = False)
    summary_corrs['T-VIKOR'] = rank
    summary_corrs = summary_corrs.rename_axis('Country')
    summary_corrs.to_csv('./results/summary.csv')

    results_final = pd.DataFrame(index = country_names)
    results_final['Temporal VIKOR pref'] = final_S
    results_final['Temporal VIKOR rank'] = rank
    results_final = results_final.rename_axis('Country')
    results_final.to_csv('./results/results_final.csv')
    

    # ===================================================================
    # Correlations
    # correlations for PLOT
    method_types = list(summary_corrs.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_rs = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(corrs.pearson_coeff(summary_corrs[i], summary_corrs[j]))
            dict_new_heatmap_rs[j].append(corrs.spearman(summary_corrs[i], summary_corrs[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$Pearson$')

    # correlation matrix with rs coefficient
    draw_heatmap(df_new_heatmap_rs, r'$r_s$')


if __name__ == '__main__':
    main()