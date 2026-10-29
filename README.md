# dkl-kan-bo

Original ORC optimization code using GP, DKL-MLP, and DKL-KAN surrogates,
together with the supplied Aspen Plus model and initial dataset.

## Files

| File | Description |
| --- | --- |
| `Gpmodel.py` | Standard GP surrogate |
| `NNGPmodel.py` | MLP feature extractor with a GP output layer |
| `KANGPmodel.py` | KAN feature extractor with a GP output layer |
| `kan1.py` | Local KAN implementation |
| `ORC_DKL_BO.ipynb` | Data preparation, candidate generation, BO, repeated trials, and plots |
| `yanzheng_KANbiaodashi.ipynb` | Aspen evaluation of a specified operating point |
| `ORC1_rundata.bkp` | Aspen Plus ORC model loaded by both notebooks |
| `ORC_LHS_Simulation_inistial_data100.xlsx` | Initial dataset read by the optimization notebook |

## Run

Use a Windows Python environment with Aspen Plus installed. The original
notebooks use the COM identifier `Apwn.Document.38.0`; adjust it if necessary
for the installed Aspen version.

```bash
pip install -r requirements.txt
```

Open this folder in VS Code or Jupyter, keeping the working directory here so
the original relative file paths resolve. Run `ORC_DKL_BO.ipynb` cell by cell.
The `run_bo_once` function currently instantiates `NNGP`; replace that class
with `BaseGP` or `KANGP` to use the other supplied surrogates. The repeated
trial settings are `N_RUNS`, `N_ITERS`, and `BATCH_SIZE_NEXT` in the notebook.

For single-point evaluation, open `yanzheng_KANbiaodashi.ipynb` and set `x1`,
`x2`, and `x3`. It evaluates the point through Aspen; it does not extract a
symbolic expression.

## Packaging notes

The four Python source files, Aspen files, and initial workbook are unchanged
from the supplied archive. Both notebooks retain their original code cells;
only saved outputs and execution counts have been cleared. Generated result
workbooks, CSV files, images, and Python caches are excluded.

The original manual candidate insertion and process shutdown cell are also
retained. The final cell of `ORC_DKL_BO.ipynb` terminates an `AspenPlus.exe`
process. Source behavior and numerical settings have not been refactored or
retuned. Actual Aspen execution has not been verified in this environment.

`requirements.txt` lists the packages imported by the original files. Exact
package versions were not supplied. Third-party KAN attribution is provided
in `THIRD_PARTY_NOTICES.md`, with its license in `LICENSES/`.
