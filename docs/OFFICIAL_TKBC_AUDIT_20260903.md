# Official TKBC Compatibility Audit

Date: 2026-09-03 (Asia/Shanghai)

## Audited source

- Repository: `facebookresearch/tkbc`
- Commit: `8cf1934d3c0106bb698683d9094bcfad93bbfc56`
- Repository license: CC BY-NC 4.0
- Data archive: the URL pinned in the repository's download script

The audit used the repository code externally. No CC BY-NC source file is
copied into this MIT-licensed project.

## Reproduction recipe in the source repository

The repository README describes TNTComplEx as its reported model for ICEWS14
and ICEWS05-15. The supplied learner uses all-entity cross-entropy, Adagrad
with learning rate 0.1, N3 embedding regularization, and Lambda3 temporal
regularization. Its documented dataset-specific settings are rank 156 with
embedding/time regularization `1e-2/1e-2` for ICEWS14 and rank 128 with
`1e-3/1` for ICEWS05-15.

## Split audit

After running the repository's unmodified ICEWS processor, the temporal support
of each split was:

| Dataset | Split | Rows | Minimum time ID | Maximum time ID | Unique time IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| ICEWS14 | train | 72,826 | 0 | 364 | 365 |
| ICEWS14 | valid | 8,941 | 0 | 364 | 365 |
| ICEWS14 | test | 8,963 | 0 | 364 | 365 |
| ICEWS05-15 | train | 368,962 | 0 | 4,016 | 4,017 |
| ICEWS05-15 | valid | 46,275 | 0 | 4,016 | 4,010 |
| ICEWS05-15 | test | 46,092 | 0 | 4,016 | 4,012 |

Thus the official partitions are not a strict future-time split. Training sees
examples from time IDs that also occur in validation and test.

## Compatibility conclusion

Official TNTComplEx learns one free embedding for every discrete time ID. Under
this project's strict chronological protocol, calibration and test contain
future time IDs absent from training, so their official embeddings would be
untrained. Using the official random-within-time split would abandon the
prequential future-time estimand; training those future embeddings would leak
future-time information; replacing them with a continuous-time map would no
longer be the official model.

For these reasons, an exact official TNTComplEx result is not inserted as a
like-for-like baseline. The paper must instead:

1. call the implemented second scorer a matched-protocol continuous-time
   complex scorer, not TComplEx or TNTComplEx;
2. state that the two scorer families are controlled protocol variants rather
   than reproductions of published TKG systems;
3. identify evaluation with architectures that extrapolate to unseen times as
   future work; and
4. avoid any claim that the evidence establishes backbone-independent
   generality.
