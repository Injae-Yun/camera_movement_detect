"""Main entry: select a detection algorithm and run on a video.

Usage examples:
  python main.py --model diff
  python main.py --model optical --display
  python main.py --model all
"""
import argparse
from utils.algorithms import available_methods

DEFAULT_VIDEO = 'data/signal_test2.mp4'


def main():
    parser = argparse.ArgumentParser(description='Run velocity detection on a video with selectable algorithm')
    parser.add_argument('--model', choices=list(available_methods().keys())+['all'], default='diff', help='Which algorithm to run')
    parser.add_argument('--video', default=DEFAULT_VIDEO, help='Path to video file')
    parser.add_argument('--display', action='store_true', help='Show detection visualizations')
    args = parser.parse_args()

    methods = available_methods()

    to_run = methods.keys() if args.model == 'all' else [args.model]

    for name in to_run:
        func = methods.get(name)
        if func is None:
            print(f"Unknown method: {name}")
            continue
        print(f"Running {name} on {args.video} (display={args.display})")
        res = func(args.video, display=args.display)
        if isinstance(res, dict) and 'error' in res:
            print(f"{name}: ERROR: {res['error']}")
            continue
        print(f"{name}: avg_v={res.get('avg_velocity',0):.2f}, count={res.get('count',0)}, runtime={res.get('runtime',0):.2f}s")


if __name__ == '__main__':
    main()
