# ICEWS14 data

Raw and processed data are intentionally excluded from Git.

Prepare the registered RE-GCN archive with:

```powershell
python scripts/prepare_icews14.py --config configs/icews14_pilot.yaml
```

The command writes `train.txt`, `valid.txt`, `test.txt`, and
`dataset_manifest.json` under `data/raw/icews14`. It refuses to replace an
existing directory and records SHA-256 for the archive and every extracted
file.
