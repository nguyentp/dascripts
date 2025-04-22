from typing import Union, List, Optional
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def decorate(
    ax,
    xaxis_dt_format: Optional[str] = None,
    xaxis_rotation: Optional[int] = None,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    title_font_size: int = 16,
    axis_font_size: int = 12,
    legend_location: Optional[str] = "best",
    legend_bbox_to_anchor: Optional[tuple] = None,
    legend_ncol: int = 1,
    figsize: tuple = (4, 4),
    grid: str = "xy",
    despine: str = "top,right",
):
    """Decorates a matplotlib axis with common styling and formatting options.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to decorate.
    xaxis_dt_format : str, optional
        DateTime format for x-axis. Options:
        - 'D': Daily format (%Y-%m-%d)
        - 'M': Monthly format (%Y-%m)
        - 'Y': Yearly format (%Y)
    xaxis_rotation : int, optional
        Rotation angle for x-axis labels in degrees.
    title : str, optional
        Title of the plot.
    xaxis_title : str, optional
        Title for the x-axis.
    yaxis_title : str, optional
        Title for the y-axis.
    title_font_size : int, default=16
        Font size for the plot title.
    axis_font_size : int, default=12
        Font size for axis labels and tick labels.
    legend_location : str, default="best"
        Location of the legend. Set to None or empty string to remove legend.
        See matplotlib's legend documentation for possible values.
    legend_bbox_to_anchor : tuple, optional
        Fine-tuning of the legend position (x, y). For example, 
        (1.05, 1) would place the legend slightly to the right of the figure.
    legend_ncol : int, default=1
        Number of columns in the legend.
    figsize : tuple, default=(4, 4)
        Figure size in inches (width, height).
    grid : str, default="xy"
        Controls grid lines:
        - 'x': Show vertical grid lines
        - 'y': Show horizontal grid lines
        - 'xy': Show both
        - '': Hide grid lines
    despine : str, default="top,right"
        Comma-separated list of spines to remove. Possible values: "top", "right", "left", "bottom"
        Set to empty string "" to keep all spines.

    Notes
    -----
    This function applies common styling to matplotlib plots including:
    - DateTime formatting for x-axis
    - Grid styling
    - Font sizes
    - Legend positioning
    - Figure size adjustment
    - Spine (border) customization
    """
    import matplotlib.dates as mdates
    DAY_FORMAT = mdates.DateFormatter("%Y-%m-%d")
    DAY_LOCATOR = mdates.DayLocator()
    MONTH_FORMAT = mdates.DateFormatter("%Y-%m")
    MONTH_LOCATOR = mdates.MonthLocator()
    YEAR_FORMAT = mdates.DateFormatter("%Y")
    YEAR_LOCATOR = mdates.YearLocator()

    if xaxis_dt_format:
        if xaxis_dt_format == "D":
            ax.xaxis.set_major_formatter(DAY_FORMAT)
            ax.xaxis.set_major_locator(DAY_LOCATOR)

        if xaxis_dt_format == "M":
            ax.xaxis.set_major_formatter(MONTH_FORMAT)
            ax.xaxis.set_major_locator(MONTH_LOCATOR)

        if xaxis_dt_format == "Y":
            ax.xaxis.set_major_formatter(YEAR_FORMAT)
            ax.xaxis.set_major_locator(YEAR_LOCATOR)

    if xaxis_rotation:
        ax.xaxis.set_tick_params(rotation=xaxis_rotation)

    if title:
        ax.set_title(title)
        
    # Set axis titles if provided
    if xaxis_title:
        ax.set_xlabel(xaxis_title)
    if yaxis_title:
        ax.set_ylabel(yaxis_title)

    # Modify legend handling to check for empty string or None
    if legend_location is None or legend_location == "":
        ax.legend().remove()
    else:
        # Use bbox_to_anchor for fine-tuning legend position if provided
        if legend_bbox_to_anchor:
            ax.legend(loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor, 
                     ncol=legend_ncol, borderaxespad=0.)
        else:
            ax.legend(loc=legend_location, ncol=legend_ncol)

    ax.title.set_fontsize(title_font_size)
    for item in (
        [ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(axis_font_size)

    ax.grid(False)  # Disable default grid lines
    if "x" in grid:
        ax.xaxis.grid(color="gray", linewidth=0.75, linestyle="--", alpha=0.8)
    if "y" in grid:
        ax.yaxis.grid(color="gray", linewidth=0.75, linestyle="--", alpha=0.8)
        
    # Handle despine option
    if despine:
        # Split and clean up the input string
        spines_to_remove = [s.strip() for s in despine.split(",") if s.strip()]
        # Remove specified spines
        for spine in spines_to_remove:
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)

    ax.figure.set_size_inches(figsize)
    ax.figure.tight_layout()
    return ax


def _default_subplot():
    fig, ax = plt.subplots()
    fig.set_size_inches((4, 4))
    return ax


def _format_numeric_value(value_format, y_val):
    try:
        # Try to format with the provided format string
        formatted_value = f"{y_val:{value_format}}"
    except ValueError:
        # If that fails (e.g., using 'd' with float), convert to int first
        if 'd' in value_format:
            formatted_value = f"{int(y_val):{value_format}}"
        else:
            # If another error, fall back to default formatting
            formatted_value = f"{y_val:.2f}"
    return formatted_value


def bar(x: Union[pd.Series, NDArray, List], y: Union[pd.Series, NDArray, List], ax=None,
        show_value: bool=True, value_fontsize: int=10, value_color: str='black', value_format: str=',.2f', 
        bar_width: float=0.8, **kwargs):
    """
    Plot a bar chart using seaborn.
    
    Based on seaborn.barplot: https://seaborn.pydata.org/generated/seaborn.barplot.html
    
    Parameters
    ----------
    x : array-like
        The x coordinates or categories for each bar.
    y : array-like
        The heights of each bar.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.
    show_value : bool, default True
        Whether to display the values on top of each bar.
    value_fontsize : int, default 10
        Font size for the value labels.
    value_color : str, default 'black'
        Color for the value labels.
    value_format : str, default ',.2f'
        Format string for the value labels.
    bar_width : float, default 0.8
        Width of the bars relative to the category width.
    **kwargs : 
        Additional arguments to pass to seaborn.barplot.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    
    # Create axis if not provided
    if ax is None:
        ax = _default_subplot()
    
    # Create the bar plot
    bars = sns.barplot(x=x, y=y, ax=ax, width=bar_width, edgecolor=None, **kwargs)
    
    # Add value labels on top of bars if requested
    if show_value:
        # Get the patches (bars) from the plot
        for i, patch in enumerate(ax.patches):
            # Get the height of the bar
            if isinstance(patch, mpatches.Rectangle):  # Ensure patch is a Rectangle
                height = patch.get_height()
            else:
                height = 0  # Default to 0 if not a Rectangle
            
            # Format the value according to specified format
            formatted_value = _format_numeric_value(value_format, height)
            
            # Add a text label on top of each bar
            ax.text(
                patch.get_x() + patch.get_width() / 2,  # X position (center of bar)
                height / 2,  # Y position (top of bar)
                formatted_value,  # Text to display
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=value_fontsize,
                color=value_color
            )
    
    # Return the axis for further customization
    return ax


def line(x: Union[pd.Series, NDArray, List], y: Union[pd.Series, NDArray, List], ax=None,
         marker: str='o', marker_color: str=None, line_color: str=None, line_width: float=1.5,
         line_style: str='-', show_value: bool=False, value_fontsize: int=8, value_format: str=',.2f',
         value_color: str="black", value_position: str='top', position_offset_x: float=0.1, position_offset_y: float=0.1, **kwargs):
    """
    Plot a line chart using seaborn.
    
    Based on seaborn.lineplot: https://seaborn.pydata.org/generated/seaborn.lineplot.html
    
    Parameters
    ----------
    x : array-like
        The x coordinates for each point.
    y : array-like
        The y coordinates for each point.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.
    marker : str, default 'o'
        The marker style for each point.
    line_color : str, default None
        The color of the line. If None, uses the default matplotlib color palette.
    marker_color : str, default None
        The color of the markers. If None, uses the line_color or default palette.
    line_width : float, default 1.5
        Width of the line. If 0, the line is not displayed (only markers show).
    line_style : str, default '-'
        Style of the line. Options include '-', '--', '-.', ':' or '' (no line).
    show_value : bool, default False
        Whether to display the values at each point.
    value_fontsize : int, default 8
        Font size for the value labels.
    value_format : str, default ',.2f'
        Format string for the value labels.
    value_position : str, default 'top'
        Position of the labels relative to markers: 'top', 'bottom', 'left', 'right'.
    position_offset_x : float, default 0.1
        Horizontal offset for value labels.
    position_offset_y : float, default 0.1
        Vertical offset for value labels.
    **kwargs : 
        Additional arguments to pass to seaborn.lineplot.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    
    # Create axis if not provided
    if ax is None:
        ax = _default_subplot()
    
    # Store original x values
    x_original = x
    
    # Create a range for x-axis positioning
    x_positions = list(range(len(x)))
    
    # Determine if we should show the line based on line_width
    # Use the provided line_style unless line_width is 0
    linestyle = '' if line_width == 0 else line_style
    
    # Create the line plot using the positions as x-coordinates
    # Let seaborn/matplotlib handle colors automatically if not specified
    ax = sns.lineplot(x=x_positions, y=y, ax=ax, marker=marker, color=line_color,
                      markerfacecolor=marker_color, markeredgecolor='none',
                      linewidth=line_width if line_width > 0 else None,
                      linestyle=linestyle, **kwargs)
    
    # Set the x-tick labels to the original x values only where y has value
    valid_positions = []
    valid_labels = []
    
    for i, (x_pos, x_orig, y_val) in enumerate(zip(x_positions, x_original, y)):
        # Only include positions where y has a valid value (not NaN or None)
        if pd.notna(y_val):
            valid_positions.append(x_pos)
            valid_labels.append(x_orig)
    
    # Set the ticks to show only at positions with valid y values
    ax.set_xticks(valid_positions)
    ax.set_xticklabels(valid_labels)
    
    # Add value labels if requested
    if show_value:
        # Determine the vertical alignment based on value_position
        va = 'bottom' if value_position == 'top' else 'top'
        ha = 'center'
        # Add text labels for each point using x_positions for positioning
        for i, (x_pos, y_val) in enumerate(zip(x_positions, y)):
            if pd.notna(y_val):  # Only add labels for valid values
                formatted_value = _format_numeric_value(value_format, y_val)
                ax.text(
                    x_pos + position_offset_x,  # X position using the range index
                    y_val + position_offset_y,  # Y position
                    formatted_value,   # Text to display
                    ha=ha,             # Horizontal alignment
                    va=va,             # Vertical alignment
                    fontsize=value_fontsize,
                    color=value_color
                )
    
    # Return the axis for further customization
    return ax


def hist(x: Union[pd.Series, NDArray, List], ax=None, bins: Union[int, List, str] = 'auto',
         kde: bool = False, stat: str = 'count', show_stats: bool = False,
         mean_line: bool = False, median_line: bool = False,
         line_color: str = 'red', line_style: str = '--', line_width: float = 1.5,
         stats_format: str = ',.2f', stats_position: str = 'bottom-center', 
         stats_fontsize: int = 10, stats_offset_y: float = -0.3,
         show_percentiles: bool = False, **kwargs):
    """
    Plot a histogram chart using seaborn.
    
    Based on seaborn.histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html
    
    Parameters
    ----------
    x : array-like
        The values to plot.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.
    bins : int, list, or str, default 'auto'
        Specification of histogram bins. Options include:
        - An int giving the number of bins
        - A list or array of bin edges
        - A string like 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'
    kde : bool, default False
        Whether to overlay a Gaussian kernel density estimate.
    stat : str, default 'count'
        Statistic to compute for the histogram. Options include:
        'count', 'percent'
    show_stats : bool, default False
        Whether to display statistics (mean, median, std) on the plot.
    mean_line : bool, default False
        Whether to show a vertical line at the mean.
    median_line : bool, default False
        Whether to show a vertical line at the median.
    line_color : str, default 'red'
        Color for mean and median lines.
    line_style : str, default '--'
        Style for mean and median lines.
    line_width : float, default 1.5
        Width for mean and median lines.
    stats_format : str, default ',.2f'
        Format string for the statistics values (mean, median, std).
    stats_position : str, default 'bottom-center'
        Position to display statistics text: 'top-right', 'bottom', or 'bottom-center'.
    stats_fontsize : int, default 10
        Font size for the statistics text.
    stats_offset_y : float, default -0.3
        Vertical offset for statistics text when positioned at bottom. Negative values 
        move the text downward (away from the plot).
    show_percentiles : bool, default False
        Whether to include percentiles (10th, 25th, 75th, 99th) in the statistics display.
    **kwargs : 
        Additional arguments to pass to seaborn.histplot.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    import numpy as np
    
    # Create axis if not provided
    if ax is None:
        ax = _default_subplot()
    
    # Create the histogram
    sns.histplot(x=x, bins=bins, kde=kde, stat=stat, ax=ax, edgecolor=None, **kwargs)
    
    # Calculate basic statistics
    mean_val = np.nanmean(x)
    median_val = np.nanmedian(x)
    std_val = np.nanstd(x)
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    
    # Calculate percentiles if needed or show_percentiles is true
    p1 = np.nanpercentile(x, 1)
    p25 = np.nanpercentile(x, 25)
    p75 = np.nanpercentile(x, 75)
    p99 = np.nanpercentile(x, 99)
    
    # Format all the statistical values
    mean_str = _format_numeric_value(stats_format, mean_val)
    median_str = _format_numeric_value(stats_format, median_val)
    std_str = _format_numeric_value(stats_format, std_val)
    min_str = _format_numeric_value(stats_format, min_val)
    max_str = _format_numeric_value(stats_format, max_val)
    p1_str = _format_numeric_value(stats_format, p1)
    p25_str = _format_numeric_value(stats_format, p25)
    p75_str = _format_numeric_value(stats_format, p75)
    p99_str = _format_numeric_value(stats_format, p99)
    
    # Add mean line if requested
    if mean_line:
        ax.axvline(float(mean_val), color=line_color, linestyle=line_style, 
                  linewidth=line_width, label=f'Mean: {mean_str}')
    
    # Add median line if requested
    if median_line:
        ax.axvline(float(median_val), color=line_color, linestyle=':', 
                  linewidth=line_width, label=f'Median: {median_str}')
    
    # Show statistics text if requested
    if show_stats:
        # Format statistics in the requested layout
        # Line 1: mean, std
        line1 = f"mean: {mean_str} | std: {std_str}"
        
        # Line 2: min, max
        line2 = f"min: {min_str} | max: {max_str}"
        
        # Line 3, 4: percentiles (only shown if show_percentiles is True)
        line3 = f"25%: {p25_str} | median: {median_str} | 75%: {p75_str}"
        line4 = f"1%: {p1_str} | 99%: {p99_str}"
        
        # Combine the lines as needed
        if show_percentiles:
            stats_text = f"{line1}\n{line2}\n{line3}\n{line4}"
        else:
            stats_text = f"{line1}\n{line2}"
        
        # Default is bottom-center
        text_x = 0.5
        text_y = stats_offset_y
        va = 'top'
        ha = 'center'
        transform = ax.transAxes
        bbox = None
        if stats_position == 'top-right':
            # Place the text box in the upper right corner
            text_x = 0.95
            text_y = 0.95
            va = 'top'
            ha = 'right'
            transform = ax.transAxes
            bbox = dict(boxstyle='round', facecolor='white', alpha=0.7)
        elif stats_position == 'bottom-center':
            # Place below the plot, centered
            text_x = 0.5
            text_y = stats_offset_y
            va = 'top'
            ha = 'center'
            transform = ax.transAxes
            bbox = None
        elif stats_position == 'bottom':
            # Place below the plot, left-aligned
            text_x = 0.05
            text_y = stats_offset_y
            va = 'top'
            ha = 'left'
            transform = ax.transAxes
            bbox = None
        
        ax.text(text_x, text_y, stats_text, transform=transform,
                verticalalignment=va, horizontalalignment=ha,
                fontsize=stats_fontsize, bbox=bbox)
        
        # Adjust the bottom margin if stats are shown below the plot
        if stats_position in ['bottom', 'bottom-center']:
            # Calculate appropriate bottom margin based on the offset and number of lines
            bottom_margin = 0.2
            if stats_offset_y < -0.15:
                bottom_margin = 0.25
            if stats_offset_y < -0.25:
                bottom_margin = 0.3
            
            # Add more space for the additional lines of statistics
            num_lines = 2  # Default (mean/std, min/max)
            if show_percentiles:
                num_lines = 3  # Add an extra line for percentiles
                
            # Adjust margin based on number of lines
            bottom_margin += (num_lines - 1) * 0.05
            
            plt.subplots_adjust(bottom=bottom_margin)
    
    # Add legend if we have lines with labels
    if mean_line or median_line:
        ax.legend()
    
    # Return the axis for further customization
    return ax


def scatter(x: Union[pd.Series, NDArray, List], y: Union[pd.Series, NDArray, List], ax=None,
           marker: str='o', size: float=50, color: Optional[str]=None, alpha: float=0.7, **kwargs):
    """
    Plot a scatter chart using seaborn.
    
    Based on seaborn.scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    
    Parameters
    ----------
    x : array-like
        The x coordinates for each point.
    y : array-like
        The y coordinates for each point.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.
    marker : str, default 'o'
        The marker style for each point.
    size : float, default 50
        Size of the markers.
    color : str, default None
        Color of the markers. If None, uses the default matplotlib color palette.
    alpha : float, default 0.7
        Opacity of the markers (0 to 1).
    **kwargs : 
        Additional arguments to pass to seaborn.scatterplot.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    # Create axis if not provided
    if ax is None:
        ax = _default_subplot()
    
    # Create the scatter plot with edgecolor set to none to remove marker edges
    sns.scatterplot(x=x, y=y, ax=ax, marker=marker, s=size, color=color, 
                   alpha=alpha, edgecolor='none', **kwargs)
    
    # Return the axis for further customization
    return ax