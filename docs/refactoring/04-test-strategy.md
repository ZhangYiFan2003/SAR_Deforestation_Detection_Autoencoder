# Test Strategy

The suite uses generated CPU-only TIFF/geospatial fixtures and does not download
Sentinel-1 data or train the full network.

Coverage includes:

- VV/VH normal, equal-count mismatch, missing VH, duplicate key, and wrong-date
  pairing;
- transform normalization, explicit clamp semantics, shape/channel validation,
  HWC-to-CHW conversion, and train/inference tensor parity;
- connected-component boundary behavior and binary metric formulas;
- GeoTIFF -> Dataset -> AE reconstruction shape;
- fixed-AOI regression where a second GT region remains an FN outside prediction
  extent;
- atomic GeoTIFF commit, manifest output, quarantine/reject path, and immutable
  raw fusion inputs;
- legacy and structured checkpoint loading;
- explicit CLI action and inference completeness status contracts;
- legacy-default and experimental model activation wiring.

GitHub Actions performs syntax compilation and the unit/CPU integration suite.
Full data training, CUDA, historical metrics, and upstream CuSum/pyroSAR remain
outside CI.
