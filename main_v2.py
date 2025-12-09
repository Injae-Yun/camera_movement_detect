"""Compatibility wrapper: call optical flow implementation from utils."""
from utils.algorithms import optical_flow_velocity

if __name__ == '__main__':
    video = 'data/signal_test2.mp4'
    print(f"Running optical flow on {video}")
    res = optical_flow_velocity(video, display=True)
    if isinstance(res, dict) and 'error' in res:
        print('ERROR:', res['error'])
    else:
        print(f"Result: avg_v={res.get('avg_velocity',0):.2f}, count={res.get('count',0)}, runtime={res.get('runtime',0):.2f}s")
