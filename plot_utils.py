import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_bytes(x, pos=None, suffix=""):
    # Human-readable byte formatter
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(x) < 1024.0:
            return f"{x:3.1f} {unit}{suffix}"
        x /= 1024.0
    return f"{x:.1f} PiB{suffix}"

def format_time(y, pos=None):
    if y == 0:
        return "0"
    elif y < 1e-3:
        return f"{y*1e6:.0f} ns"
    elif y < 1:
        return f"{y*1e3:.1f} μs"
    elif y < 1e3:
        return f"{y:.2f} ms"
    else:
        return f"{y/1e3:.2f} s"

def plot_latency_results(results, output_file=None, gpu_name=None):
    sizes = results['size_bytes']
    write_means = results['write_ms_mean']
    write_stds = results['write_ms_std']
    read_means = results['read_ms_mean']
    read_stds = results['read_ms_std']
    write_bw_means = results['write_bw_mean']
    write_bw_stds = results['write_bw_std']
    read_bw_means = results['read_bw_mean']
    read_bw_stds = results['read_bw_std']

    fig, ax1 = plt.subplots(figsize=(10, 6))
    l1 = ax1.errorbar(sizes, write_means, yerr=write_stds, label='Write Latency', fmt='-o', color='tab:blue')
    l2 = ax1.errorbar(sizes, read_means, yerr=read_stds, label='Read Latency', fmt='-o', color='tab:orange')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_bytes(x, pos)))
    ax1.xaxis.set_major_locator(ticker.LogLocator(base=2))
    ax1.xaxis.set_minor_locator(ticker.LogLocator(base=2, subs='auto', numticks=100))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.grid(True, which='major', axis='x', ls='-', lw=1, alpha=0.7)
    ax1.grid(True, which='minor', axis='x', ls=':', lw=0.1, alpha=0.5)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    ax1.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax1.set_xlabel('Tensor Size')
    ax1.set_ylabel('Latency')
    ax2 = ax1.twinx()
    l3 = l4 = None
    if len(write_bw_means) > 0:
        l3 = ax2.errorbar(sizes, [x * 1024**3 for x in write_bw_means], yerr=[x * 1024**3 for x in write_bw_stds], label='Write Bandwidth', fmt='--s', color='tab:green')
    if len(read_bw_means) > 0:
        l4 = ax2.errorbar(sizes, [x * 1024**3 for x in read_bw_means], yerr=[x * 1024**3 for x in read_bw_stds], label='Read Bandwidth', fmt='--s', color='tab:red')
    ax2.set_yscale('log', base=2)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_bytes(y, pos, suffix='/s')))
    ax2.yaxis.set_major_locator(ticker.LogLocator(base=2))
    ax2.yaxis.set_minor_locator(ticker.LogLocator(base=2, subs='auto', numticks=100))
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.grid(True, which='minor', axis='y', ls=':', lw=0.7, alpha=0.5)
    ax2.set_ylabel('Bandwidth (bytes/sec)')
    lines, labels = ax1.get_legend_handles_labels()
    if l3 is not None or l4 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper left')
    title = 'GPU Memory Latency & Bandwidth'
    if gpu_name:
        title += f' ({gpu_name})'
    plt.title(title)
    ax1.grid(True, which="both", axis='y', ls="--", lw=0.5)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
