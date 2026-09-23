# Entry-ending run aborted before results

The driver at fd3f6b2 raised `NameError: np is not defined` while assembling the first
control score, before printing or saving any metric. No score informed the repair.
The frozen code remains unchanged; the import-only correction, new freeze and results
are in [v2](../linear-a-structure-v2/REPORT.md).
