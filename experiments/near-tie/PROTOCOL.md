# Protocol for the one-or-two reason rule

This protocol was written and committed before the fresh splits were run.
Seeds 20261027–20261031 were exploratory only. The fresh evaluation uses
20261032–20261036.

## Definitions

Group score is the absolute signed sum. For each source fit, `m1` and `m2` are
the largest and second-largest scores; ties keep partition order. The relative
gap is `(m1 - m2) / m1`; an all-zero row is excluded and counted. Each case is
one of 90 ordered pairs from ten fits. The source fit alone decides what the
borrower sees. The second fit is the replicate.

## Rule frozen from exploratory data

Show one reason when its relative gap is at least 0.36;
otherwise show the leading two groups. This threshold was selected as the
smallest grid value whose pooled single-reason repeat rate is at least 90 percent. On exploratory cases it showed one reason
in 71.6% of cases, with 90.3%
repeat agreement.

## Fresh measures and controls

For each fresh split report: one-reason coverage, its repeat rate `R1`,
two-reason rate, containment of the replicate first reason, and all-case
repeat rate `B`. The primary same-coverage comparison is `R1 - B`, because a
random subset at the same coverage has expected repeat rate `B`.

Also report containment against (1) a random subset receiving two reasons and
(2) the leading reason plus a uniformly random other group. Report selective
repeat rates at 25, 50, 75 and 100 percent coverage for correlation groups
and the mean of the 30 size-matched random partitions.

The rule is a meaningful gain only if `R1 - B` is at least 5 percentage points
at coverage at least 50 percent on at least 4 of the 5 fresh splits. Top-two
containment alone is not an improvement claim. Fresh splits reuse one source
population and can share borrowers, so results are descriptive.

The audit seed remains 20; bootstrap seeds remain 0–9; random partitions
remain the 30 draws from seed 101.
