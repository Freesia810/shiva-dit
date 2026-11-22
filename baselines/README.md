# Comparison methods

The comparison hooks run in the same framework as Shiva-DiT. Appendix B.4 gives the settings for the paper's comparisons. [Paper](https://openreview.net/forum?id=uAAaT1KajI)

| Method | Hook | Assets |
| --- | --- | --- |
| ToMeSD | [tome_sd.py](tome_sd.py) | [Example configuration](../benchmark/meta_config/tome_sd.json) · [Upstream](https://github.com/dbolya/tomesd) |
| ToFu | [tofu.py](tofu.py) | [Example configuration](../benchmark/meta_config/tofu.json) |
| SDTM | [sdtm.py](sdtm.py) | [Example configuration](../benchmark/meta_config/sdtm.json) · [Upstream](https://github.com/ICTMCG/SDTM) |
| ToMA | [toma.py](toma.py) | [Example configuration](../benchmark/meta_config/toma.json) |
| IBTM | [ibtm.py](ibtm.py) | [Example configuration](../benchmark/meta_config/ibtm.json) · [Upstream](https://github.com/cvlab-stonybrook/importance-token-merge) |
| DiffCR | [diffcr.py](diffcr.py) | Method-specific checkpoint · [Upstream](https://github.com/GATECH-EIC/DiffCR) |
| DyDiT | [dy_dit.py](dy_dit.py) | Method-specific checkpoint |
| SparseDiT | [sparse_dit.py](sparse_dit.py) | Method-specific checkpoint |
| NeuralSort | [neuralsort.py](neuralsort.py) | Method-specific checkpoint |

Pass `--meta_config benchmark/meta_config/tome_sd.json` to the generation or timing command to run ToMeSD with the corresponding base model. For a learned baseline, pass its compatible `--ckpt_path` instead. Use the same prompt list and sampling protocol as the Shiva configuration being compared.
