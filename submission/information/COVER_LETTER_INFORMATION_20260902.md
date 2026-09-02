# Cover Letter

2 September 2026

Dear Editors of *Information*,

Please consider the manuscript **“Prequential Answer-Set Calibration for Temporal Knowledge Graph Forecasting”** for publication as an Article in *Information*.

Temporal knowledge graph forecasting is commonly evaluated with ranking metrics, although downstream use often requires a set of candidate entities with a stated empirical coverage level. This manuscript studies whether an early static calibration threshold remains reliable after a chronological split, and whether a timestamp-batched rolling threshold offers a useful repair. Its contribution is a leakage-controlled prequential evaluation and a reliability--utility diagnostic. It does not claim a new distribution-free theorem under arbitrary temporal drift.

The confirmatory study covers ICEWS14 and ICEWS05-15, temporal DistMult and a matched-protocol continuous-time complex scorer, five training seeds, controlled training-fact deletion, and filtered-versus-uniform negative-sampling sensitivity. It compares rolling and static margin calibration with three published KGCP nonconformity transformations under a common static split. All 90 planned seed--deletion conditions completed. At 30% deletion and a 0.90 target, rolling observed-label coverage is 0.8988--0.9002 across the four dataset--scorer combinations. The comparison with KGCP Minmax and NegScore is heterogeneous, and rolling prediction sets are often substantially larger. The manuscript also reports a pronounced multi-answer full-set coverage deficit and null or inconclusive deletion and negative-sampling effects.

The manuscript is original, is not under consideration elsewhere, and has been approved by the author. The study uses public benchmark data and requires no institutional review board approval or informed consent. Source code, resolved configurations, derived result tables, figure-generation scripts, checksums, and a clean-clone release audit are available under the MIT License at https://github.com/xywang815/riskcal-tkg-w. The submitted artifact snapshot is identified by the tag `v1.0.0-mdpi-information-submission`.

Thank you for considering this manuscript.

Sincerely,

Xinyu Wang  
Anhui Institute of Information Technology  
Wuhu 241000, China  
xywang68@iflytek.com
