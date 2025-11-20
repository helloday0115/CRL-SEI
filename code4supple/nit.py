"""nit.py
NIT utilities: sample Bézier control points and apply amplitude-envelope warping to 1D signals.
Integrate into dataset augmentation pipeline: for each signal, sample control points (seeded),
compute a Bezier amplitude envelope, multiply the signal by the envelope.

Functions:
  - sample_bezier_control_points(num_ctrl=5, amp_min=0.8, amp_max=1.25, seed=None)
  - bezier_envelope_transform(signal, ctrl_points)
"""
import numpy as np

def bezier_eval(points, t):
    # De Casteljau's algorithm for scalar control points
    pts = np.array(points, dtype=float)
    t = np.array(t)
    # evaluate for vector t: apply De Casteljau iteratively
    pts_copy = pts.copy()
    n = len(points) - 1
    for r in range(1, n+1):
        pts_copy[:n-r+1] = (1 - t[:,None]) * pts_copy[:n-r+1] + t[:,None] * pts_copy[1:n-r+2]
    return pts_copy[0]

def sample_bezier_control_points(num_ctrl=5, amp_min=0.8, amp_max=1.25, seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(amp_min, amp_max, size=num_ctrl).tolist()
    # keep endpoints around 1 for smoothness
    pts[0] = 1.0
    pts[-1] = 1.0
    return pts

def bezier_envelope_transform(signal, ctrl_points):
    sig = np.asarray(signal).reshape(-1)
    T = sig.shape[0]
    t = np.linspace(0.0, 1.0, T)
    env = bezier_eval(ctrl_points, t)
    return (sig * env).astype(sig.dtype)

# quick demo
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.sin(np.linspace(0, 20*np.pi, 2048))
    cp = sample_bezier_control_points(num_ctrl=6, amp_min=0.7, amp_max=1.3, seed=0)
    y = bezier_envelope_transform(x, cp)
    print('control points:', cp)
    plt.plot(x[:400], label='orig'); plt.plot(y[:400], label='nit'); plt.legend(); plt.show()
