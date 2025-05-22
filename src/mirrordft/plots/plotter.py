from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
from transformers import HfArgumentParser
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

@dataclass
class PlotConfig:
    stamp: Optional[str] = field(default='20250309-072304')
    title: bool = field(default=True)
    file_name: Optional[str] = field(default=None)
    save_path: Optional[str] = field(default=None)

# parser = HfArgumentParser(PlotConfig)
# config = parser.parse_args_into_dataclasses()[0]
# config_dict = vars(config)
# locals().update(config_dict)

def find_stamp_folder(stamp, base_dir='./outputs'):
    for root, dirs, files in os.walk(base_dir):
        if stamp in os.path.basename(root):
            return os.path.relpath(root, start=os.getcwd())
    return None

def my_legend(axe=None,loc=0):
    if axe is None:
        axe = plt.gca()
    legend = axe.legend(loc=loc)
    legend.get_frame().set_linewidth(legend_linewidth)
    legend.get_frame().set_edgecolor('black')


plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
legend_linewidth = 1


def my_inset_plot(main_ax, x_data, y_data, optimal_value, percent=0.6,show_optimal=True):
    axins = inset_axes(main_ax, width="60%", height="60%", 
                    loc=1,
                    bbox_to_anchor=(0.00,0.0, 0.95, 0.95),
                    bbox_transform=main_ax.transAxes)

    # Plot the same data in the inset
    axins.plot(x_data, y_data, color='blue')
    if show_optimal:
        axins.hlines(optimal_value, 0, max(x_data), colors='r', linestyles='--')

    sub_y_data = y_data[int(len(y_data)*percent):] 
    max_abs_dist = max(abs(sub_y_data - optimal_value))
    axins.set_ylim(optimal_value - max_abs_dist*1.1, optimal_value + max_abs_dist*1.1)
    # Use scientific notation for y-axis ticks
    axins.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Set x-limits to show the latter part of iterations
    axins.set_xlim(max(x_data) * percent-10, max(x_data)+10)
    axins.grid(True)