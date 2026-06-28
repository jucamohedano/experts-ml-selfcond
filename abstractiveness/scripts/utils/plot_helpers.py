import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def _plot_bar_chart(
    plot_dataframe, x_col, y_col,
    title, x_label=None, y_label=None, legend_title=None,
    color=None, hue=None, palette=None, orient='v', custom_tick_labels=None,
    out_path=None, figsize=(40.56, 10.14), ax=None, save=True, **kwargs
) -> plt.Axes:
    """
    Create a bar chart with category labels on the relevant axis.
    Supports both vertical and horizontal orientations, optional hue-based grouping, and color customization.
    Returns the matplotlib Axes object, and optionally saves to file if out_path is provided.
    
    Args:
        plot_dataframe: DataFrame containing the data to plot.
        x_col: Column name for x-axis (vertical) or bars (horizontal).
        y_col: Column name for y-axis values or bar labels (horizontal).
        title: Plot title.
        x_label, y_label: Axis labels.
        legend_title: Title for legend if hue grouping is used.
        color: Single color for all bars (if no palette/hue).
        hue: Column to group bars by color.
        palette: Color palette for hue groups.
        orient: 'v' for vertical bars, 'h' for horizontal bars.
        custom_tick_labels: Optional custom labels for axes.
        out_path: Path to save the figure (PNG at 300 dpi).
        figsize: Figure size as (width, height).
        ax: Existing axes object to plot on; creates new figure if None.
        save: If True and ax is None, saves figure; always displays if ax is not None.
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax_current = plt.gca()
    else:
        ax_current = ax

    plot_args = {"edgecolor": "black", "orient": orient}
    if hue is not None:
        plot_args["hue"] = hue
        if palette is not None: plot_args["palette"] = palette
    elif palette is not None:
        plot_args["palette"] = palette
    else:
        plot_args["color"] = color
        
    plot_args.update(kwargs) 

    sns.barplot(data=plot_dataframe, x=x_col, y=y_col, ax=ax_current, **plot_args)
    
    ax_current.set_title(title, fontsize=18 if save else 14, pad=15)

    if legend_title and ax_current.get_legend():
        ax_current.legend(title=legend_title, loc='upper right' if not save else 'best')

    if orient == 'v':
        if custom_tick_labels is not None:
            labels = custom_tick_labels
        else:
            labels = [t.get_text() for t in ax_current.get_xticklabels()]
            
        ax_current.set_xticks(ax_current.get_xticks())
        ax_current.set_xticklabels(labels, rotation=45, ha='right', fontsize=10 if save else 9)
        ax_current.tick_params(axis='x', length=10, width=1, direction='out', color='black')
        
        if x_label: ax_current.set_xlabel(x_label, fontsize=14 if save else 11)
        else: ax_current.set_xlabel("")
        if y_label: ax_current.set_ylabel(y_label, fontsize=14 if save else 11)
        else: ax_current.set_ylabel("")
        
        if ax is None: plt.subplots_adjust(bottom=0.25)

    elif orient == 'h':
        if custom_tick_labels is not None:
            labels = custom_tick_labels
        else:
            labels = [t.get_text() for t in ax_current.get_yticklabels()]
            
        ax_current.set_yticks(ax_current.get_yticks())
        ax_current.set_yticklabels(labels, fontsize=12)
        ax_current.tick_params(axis='y', length=10, width=1, direction='out', color='black')

        if x_label: ax_current.set_xlabel(x_label, fontsize=14 if save else 11)
        else: ax_current.set_xlabel("")
        if y_label: ax_current.set_ylabel(y_label, fontsize=14 if save else 11)
        else: ax_current.set_ylabel("")
                        
        if ax is None: plt.subplots_adjust(left=0.25)

    if save and ax is None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    return ax_current


def _plot_heatmap_with_leaders(matrix, concepts, title, out_path, cmap) -> None:
    """
    Generate a heatmap visualization of a similarity/distance matrix with concept labels and leader lines.
    Each column is labeled with a concept name via annotated leader lines from the top edge.
    Saves the resulting figure as a high-resolution PNG.
    
    Args:
        matrix: 2D numpy array or matrix to visualize.
        concepts: List of concept names for row/column labels.
        title: Plot title.
        out_path: Path where to save the PNG file.
        cmap: Colormap name for the heatmap.
    """
    n = len(concepts)
    plt.figure(figsize=(39.5, 32.96))
    ax = sns.heatmap(matrix, xticklabels=False, yticklabels=concepts, cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("")

    for i, concept in enumerate(concepts):
        x_pos = i + 0.5 
        ax.annotate("", xy=(x_pos, n), xytext=(0, -10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
        ax.annotate(concept, xy=(x_pos, n), xytext=(3, -6), textcoords="offset points",
                    ha='right', va='top', rotation=45, fontsize=10)

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()