# fedavg-pytorch-benchmark

A minimal PyTorch implementation of [FedAvg](https://arxiv.org/abs/1602.05629) that I put together to understand how data heterogeneity affects federated learning convergence. Nothing fancy — just MNIST, a small CNN, and a clean simulation loop.

---

## What this is

I wanted to see firsthand how much IID vs Non-IID data distributions actually matter in federated setups. The setup is 10 clients, each training locally for 2 epochs per round, with 5 clients randomly selected each round. The Non-IID split gives each client only 2 digit classes (sorted shard method from the original FedAvg paper).

## Running it

```bash
pip install torch torchvision numpy matplotlib
python fedavg_simulation.py
```

MNIST downloads automatically on first run. The script trains both IID and Non-IID setups back-to-back and saves a plot to `fedavg_results.png`.

## Results

![convergence plot](fedavg_results.png)

IID converges fast and stays stable — above 98% by round 3, plateaus around 99.1% and barely moves after that.

Non-IID is a different story. It starts around 25% (basically random) and takes until round 10+ to get competitive. Even then it's noisy — swinging between 86–95% in the final rounds instead of settling. The final round accuracy was 91.67% vs 99.16% for IID, so roughly a 7.5% gap just from data heterogeneity with the same model and same number of rounds.

The core issue is client drift: when each client only sees 2 digit classes, local SGD pulls the model toward its own class distribution. After aggregation, those conflicting updates partially cancel out, which slows convergence and introduces variance. More local epochs makes it worse (checked this separately — E=5 Non-IID is noticeably worse than E=1).

## Files

- `fedavg_simulation.py` — the full simulation: model definition, data partitioning, local training, FedAvg aggregation, evaluation
- `fedavg_results.png` — convergence curves for IID vs Non-IID

## Notes

- Tested on Python 3.14 / PyTorch 2.13 / macOS (CPU only)
- If you hit an SSL error on MNIST download, run `pip install --upgrade certifi` or add `import ssl; ssl._create_unverified_context = ssl.create_default_context` at the top
- Partial participation (50% per round) is baked in — change `num_clients // 2` in `run_experiment()` to adjust

## Ablation: local epochs (E=1 vs E=2 vs E=5)

I ran a follow-up experiment sweeping local epochs to test the classic claim that more local steps = more client drift on Non-IID data.

![ablation drift](fedavg_ablation.png)

![iid vs non-iid gap](fedavg_ablation_gap.png)

Final accuracy after 15 rounds:

| | IID | Non-IID |
|---|---|---|
| E=1 | 98.94% | 94.88% |
| E=2 | 99.12% | 96.24% |
| E=5 | 99.15% | 96.29% |

On MNIST the IID–NonIID gap stays around 3–4% regardless of E, and more local epochs actually help slightly even in the Non-IID case. The dramatic drift effect from theory is more visible on harder datasets (CIFAR, Shakespeare, etc.) where label distributions are more extreme. The bigger story here is the variance — Non-IID curves oscillate round-to-round while IID ones are almost flat after round 5.

Run the ablation yourself:

```bash
python fedavg_ablation.py
```

## References

- McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629), AISTATS 2017
- Li et al., [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), MLSys 2020 (FedProx — addresses the drift problem)
