"""Run all available algorithms (non-display) and print a short comparison."""
import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.algorithms import available_methods

DEFAULT_VIDEO = 'data/signal_test2.mp4'


def _downsample_series(x, max_points=20000):
    n = len(x)
    if n <= max_points:
        return np.array(x)
    stride = max(1, n // max_points)
    return np.array(x[::stride])


def run_all(video_path=DEFAULT_VIDEO, plot=True):
    methods = available_methods()
    results = {}
    for name, func in methods.items():
        if func is None:
            print(f"Skipping {name}: not implemented")
            continue
        print(f"Running {name}...", end=' ', flush=True)
        start = time.time()
        res = func(video_path, display=False)
        elapsed = time.time() - start
        if isinstance(res, dict) and 'error' in res:
            print(f"failed: {res['error']}")
            results[name] = res
            continue
        res['measured_time'] = elapsed
        results[name] = res
        # print both sample_count and frame_count when available
        sc = res.get('sample_count', res.get('count', 0))
        fc = res.get('frame_count', None)
        if fc is not None:
            print(f"done (samples={sc}, frames={fc}, t={elapsed:.2f}s)")
        else:
            print(f"done (samples={sc}, t={elapsed:.2f}s)")

    print('\nSummary:')
    for name, r in results.items():
        if 'error' in r:
            print(f"- {name}: ERROR: {r['error']}")
        else:
            sc = r.get('sample_count', r.get('count', 0))
            fc = r.get('frame_count', None)
            runtime = r.get('runtime', r.get('measured_time', 0.0))
            if fc is not None:
                print(f"- {name}: samples={sc}, frames={fc}, runtime={runtime:.2f}s")
            else:
                print(f"- {name}: samples={sc}, runtime={runtime:.2f}s")

    # Plot and save per-method per-frame series (prefer 'per_frame' if available)
    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib not available, skipping plot: {e}")
            return results

        out_dir = os.path.dirname(__file__)
        combined_plotted = 0
        # first, save individual plots per method
        for name, r in results.items():
            if 'error' in r:
                continue
            series = r.get('per_frame') or r.get('velocities')
            if not series:
                continue
            arr = _downsample_series(series, max_points=20000)
            x = np.arange(len(arr))
            plt.figure(figsize=(10, 4))
            # small smoothing for readability
            win = max(1, min(31, len(arr)//200 if len(arr)>200 else 3))
            if win > 1:
                kernel = np.ones(win)/win
                smooth = np.convolve(arr, kernel, mode='same')
            else:
                smooth = arr
            plt.plot(x, smooth, linewidth=1)
            plt.xlabel('frame or sample index')
            plt.ylabel('movement (pixels/frame)')
            plt.title(f'{name} per-frame movement (n={len(arr)})')
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'velocities_{name}.png')
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {name} velocity plot to: {out_path}")
            combined_plotted += 1

        # also save a combined plot if multiple series available
        if combined_plotted > 1:
            plt.figure(figsize=(12, 6))
            for name, r in results.items():
                if 'error' in r:
                    continue
                series = r.get('per_frame') or r.get('velocities')
                if not series:
                    continue
                arr = _downsample_series(series, max_points=20000)
                x = np.arange(len(arr))
                win = max(1, min(31, len(arr)//200 if len(arr)>200 else 3))
                if win > 1:
                    kernel = np.ones(win)/win
                    smooth = np.convolve(arr, kernel, mode='same')
                else:
                    smooth = arr
                plt.plot(x, smooth, label=f"{name} (n={len(arr)})", linewidth=1)
            plt.xlabel('frame or sample index')
            plt.ylabel('movement (pixels/frame)')
            plt.title('Per-frame movement comparison')
            plt.legend(loc='upper right')
            plt.tight_layout()
            out_path = os.path.join(out_dir, 'velocities_combined.png')
            plt.savefig(out_path)
            plt.close()
            print(f"Saved combined velocity plot to: {out_path}")

        # Save a bar plot of runtimes per method
        runtimes = {}
        for name, r in results.items():
            if 'error' in r:
                continue
            # prefer explicit runtime, fallback to measured_time
            runtimes[name] = r.get('runtime', r.get('measured_time', 0.0))

        if runtimes:
            names = list(runtimes.keys())
            times = [runtimes[n] for n in names]
            plt.figure(figsize=(8, 4))
            plt.bar(names, times, color='C2')
            plt.ylabel('seconds')
            plt.title('Algorithm runtime (seconds)')
            for i, v in enumerate(times):
                plt.text(i, v + max(0.01, max(times)*0.01), f"{v:.2f}s", ha='center', va='bottom', fontsize=9)
            out_path = os.path.join(out_dir, 'runtimes.png')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Saved runtime barplot to: {out_path}")

    return results


if __name__ == '__main__':
    # allow PYTHONPATH caller; ensure video path resolved relative to repo
    run_all()
