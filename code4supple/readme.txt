(i) Deterministic train/val/test partitioning script (splins.py) and recommended fixed seeds list (directly reproducible); 
(ii) NIT implementation (including Bezier control point sampling + signal envelope transformation) (nit.py) and arr (Gumbel-Softmax K-hot mask) PyTorch implementation (arr.py); 
(iii) Weibull Calibration Fitting/Evaluation Script (weibull_calibrator.py); 
(iv) Minimum configuration file (Table I style, yaml) (config.yaml); 
(v) "One-click" run the script skeleton (run_all.sh) and instructions for using the README.

It is recommended to use three fixed random seeds (for training/partitioning/evaluation) in the reproduction of the paper:
SEEDS = [42, 2025, 1234]

Splins.py - Generate deterministic train/val/test partitions (ADS-B and WiFi)
Nit.py - Bezier Control Point Sampling and Signal envelope Transformation (NIT)
arr.py - Gumbel-Softmax K-hot Mask (PyTorch)
weibull_Calibrator.py - Fitting and Evaluation of Weibull calibrator (scipy)
config.yaml - Minimum Hyperparameter configuration
run_all.sh - One-click Preparation (Skeleton)
