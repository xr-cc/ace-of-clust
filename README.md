# ACE-OF-Clust (`ace-of-clust`)
ACE-OF-Clust (**A**lignment, **C**omparison, and **E**valutaion of **O**mics **F**eatures in **Clust**ing) is a small utility package built on top of [`clumppling`](https://pypi.org/project/clumppling/) to help run clustering-alignment workflows and compute downstream comparisons / summaries on single-cell omics clustering analyses.

- **PyPI name (install):** `ace-of-clust`
- **Python import (module):** `ace_of_clust`

## Installation

```bash
pip install ace-of-clust
````

Optional (better label adjustment in some plots):

```bash
pip install "ace-of-clust[adjusttext]"
```

## Quickstart

### Run `clumppling` / `compModels` via wrappers

```python
from pathlib import Path

from ace_of_clust.wrappers import (
    run_clumppling_via_main,
    prepare_comp_models_inputs,
    run_comp_models,
)

# Example: run clumppling on an existing results directory / config
res_dir = Path("output/clumppling_run")
run_clumppling_via_main(res_dir=res_dir)

# Example: prepare and run compModels (paths/args will depend on your pipeline)
input_dir = Path("output/comp_models_inputs")
prepare_comp_models_inputs(input_dir=input_dir)

run_comp_models(input_dir=input_dir, res_dir=Path("output/comp_models_results"))
```

### Analysis helpers

```python
import pandas as pd
from ace_of_clust.analysis import (
    compute_profile,
    extract_all_mode_pair_mappings,
    map_alt_to_ref,
    compute_overall_membership_difference,
)

# Example placeholders: replace with your real objects / inputs
# profile = compute_profile(...)
# mappings = extract_all_mode_pair_mappings(...)
# alt2ref = map_alt_to_ref(...)
# diff = compute_overall_membership_difference(...)
```

### Plotting

```python
from ace_of_clust.plot import (
    # add your main plotting entry points here
)

# Example:
# fig = plot_something(...)
# fig.savefig("figures/example.png", dpi=200, bbox_inches="tight")
```

## Reproducing examples

This repo keeps example scripts/notebooks separate from the installable library code.
To reproduce examples:

1. Install the package (`pip install ace-of-clust`)
2. Clone this repository (for `examples/`, `scripts/`, etc.)
3. Run the example scripts while using the installed package.

