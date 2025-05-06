"""
Plotting utilities for GPU benchmarking results.

This module provides functions for plotting bandwidth and latency results
with support for comparing multiple test functions and devices.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def format_bytes(x: float, pos: Optional[int] = None, suffix: str = "") -> str:
    """
    Format bytes into human-readable strings with appropriate units.
    
    Args:
        x: Value in bytes
        pos: Position (used by matplotlib formatter)
        suffix: Optional suffix to append (e.g., "/s" for bandwidth)
        
    Returns:
        Formatted string with appropriate unit
    """
    if x == 0:
        return f"0 B{suffix}"
    
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    base = 1024.0
    
    i = 0
    while abs(x) >= base and i < len(units) - 1:
        x /= base
        i += 1
    
    return f"{x:.1f} {units[i]}{suffix}"


def format_time(y: float, pos: Optional[int] = None) -> str:
    """
    Format time values into human-readable strings with appropriate units.
    
    Args:
        y: Time value in seconds
        pos: Position (used by matplotlib formatter)
        
    Returns:
        Formatted string with appropriate time unit
    """
    if y == 0:
        return "0"
    elif y < 1e-6:  # Less than 1 nanosecond
        return f"{y*1e9:.1f} ns"
    elif y < 1e-3:  # Less than 1 microsecond
        return f"{y*1e6:.1f} ns"
    elif y < 1:     # Less than 1 millisecond
        return f"{y*1e3:.1f} μs"
    elif y < 1e3:   # Less than 1 second
        return f"{y:.2f} ms"
    else:
        return f"{y/1e3:.2f} s"


def plot_bandwidth_latency(
    results: Dict[str, Dict[str, List[float]]],
    title: str = "Bandwidth and Latency",
    output_file: Optional[str] = None,
    log_x: bool = True,
    log_y: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    include_bandwidth: bool = True,
    include_latency: bool = True,
    operation_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot bandwidth and/or latency results for one or more operations.
    
    Args:
        results: Dictionary of results with keys for different operations
                Each operation should have 'size_bytes', 'latency_mean', 'latency_std',
                and optionally 'bandwidth_mean', 'bandwidth_std'
        title: Plot title
        output_file: If provided, save plot to this file
        log_x: Use logarithmic x-axis
        log_y: Use logarithmic y-axis
        figsize: Figure size (width, height) in inches
        include_bandwidth: Whether to plot bandwidth
        include_latency: Whether to plot latency
        operation_names: Optional list of operation names to include (if None, include all)
        
    Returns:
        Matplotlib figure object
    """
    if not include_bandwidth and not include_latency:
        raise ValueError("At least one of include_bandwidth or include_latency must be True")
    
    # Filter operations if specified
    if operation_names:
        results = {k: v for k, v in results.items() if k in operation_names}
    
    if not results:
        raise ValueError("No results to plot")
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Colors for different operations
    colors = plt.cm.tab10.colors
    
    # Line styles for latency vs bandwidth
    latency_style = '-o'
    bandwidth_style = '--s'
    
    # Set up axes
    if log_x:
        ax1.set_xscale('log', base=2)
    
    if include_latency:
        if log_y:
            ax1.set_yscale('log')
        ax1.set_ylabel('Latency (seconds)')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    
    # Create second y-axis for bandwidth if needed
    ax2 = None
    if include_bandwidth:
        if include_latency:
            ax2 = ax1.twinx()
            if log_y:
                ax2.set_yscale('log', base=2)
            ax2.set_ylabel('Bandwidth (bytes/sec)')
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, pos: format_bytes(y, pos, suffix='/s')))
        else:
            # If only plotting bandwidth, use the primary axis
            if log_y:
                ax1.set_yscale('log', base=2)
            ax1.set_ylabel('Bandwidth (bytes/sec)')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, pos: format_bytes(y, pos, suffix='/s')))
    
    # Plot each operation
    legend_handles = []
    legend_labels = []
    
    for i, (op_name, op_results) in enumerate(results.items()):
        color = colors[i % len(colors)]
        sizes = op_results['size_bytes']
        
        # Plot latency if included
        if include_latency and 'latency_mean' in op_results:
            latency_means = op_results['latency_mean']
            latency_stds = op_results.get('latency_std', [0] * len(latency_means))
            
            line = ax1.errorbar(
                sizes, latency_means, yerr=latency_stds,
                label=f'{op_name} Latency', fmt=latency_style, color=color
            )
            legend_handles.append(line)
            legend_labels.append(f'{op_name} Latency')
        
        # Plot bandwidth if included
        if include_bandwidth and 'bandwidth_mean' in op_results:
            bandwidth_means = op_results['bandwidth_mean']
            bandwidth_stds = op_results.get('bandwidth_std', [0] * len(bandwidth_means))
            
            target_ax = ax2 if ax2 is not None else ax1
            line = target_ax.errorbar(
                sizes, bandwidth_means, yerr=bandwidth_stds,
                label=f'{op_name} Bandwidth', fmt=bandwidth_style, color=color
            )
            legend_handles.append(line)
            legend_labels.append(f'{op_name} Bandwidth')
    
    # Set up x-axis
    ax1.set_xlabel('Data Size (bytes)')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    # Add grid
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    if log_x or log_y:
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.4)
    
    # Add legend
    if legend_handles:
        ax1.legend(legend_handles, legend_labels, loc='best')
    
    # Set title
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


def plot_comparison(
    results_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric: str = 'bandwidth_mean',
    title: str = "Performance Comparison",
    output_file: Optional[str] = None,
    log_x: bool = True,
    log_y: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    operation_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot a comparison of a specific metric across different test configurations.
    
    Args:
        results_dict: Nested dictionary with test configuration names as top-level keys,
                     and operation results as values
        metric: The metric to compare ('bandwidth_mean', 'latency_mean', etc.)
        title: Plot title
        output_file: If provided, save plot to this file
        log_x: Use logarithmic x-axis
        log_y: Use logarithmic y-axis
        figsize: Figure size (width, height) in inches
        operation_names: Optional list of operation names to include (if None, include all)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up axes
    if log_x:
        ax.set_xscale('log', base=2)
    if log_y:
        ax.set_yscale('log')
    
    # Colors for different test configurations
    colors = plt.cm.tab10.colors
    
    # Line styles for different operations
    line_styles = ['-o', '--s', '-.^', ':d', '-x']
    
    # Plot each test configuration and operation
    legend_handles = []
    legend_labels = []
    
    for i, (config_name, config_results) in enumerate(results_dict.items()):
        if operation_names:
            # Filter operations if specified
            config_results = {k: v for k, v in config_results.items() if k in operation_names}
        
        for j, (op_name, op_results) in enumerate(config_results.items()):
            if metric not in op_results:
                continue
                
            color = colors[i % len(colors)]
            line_style = line_styles[j % len(line_styles)]
            
            sizes = op_results['size_bytes']
            values = op_results[metric]
            stds = op_results.get(metric.replace('mean', 'std'), [0] * len(values))
            
            line = ax.errorbar(
                sizes, values, yerr=stds,
                label=f'{config_name} - {op_name}',
                fmt=line_style, color=color
            )
            legend_handles.append(line)
            legend_labels.append(f'{config_name} - {op_name}')
    
    # Set up axes labels and formatting
    ax.set_xlabel('Data Size (bytes)')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    if 'bandwidth' in metric.lower():
        ax.set_ylabel('Bandwidth (bytes/sec)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: format_bytes(y, pos, suffix='/s')))
    elif 'latency' in metric.lower() or 'time' in metric.lower():
        ax.set_ylabel('Latency (seconds)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    
    # Add grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    if log_x or log_y:
        ax.grid(True, which='minor', linestyle=':', linewidth=0.2, alpha=0.4)
    
    # Add legend
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='best')
    
    # Set title
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig
