Across the saved deterministic PPO rerun, the session cap does not materially reduce daily watch: `CumWatch` stays within 0.00% of PPO across `T_cap=90/120/150`, and at the default `T_cap=120` it moves up only from 23.1 to 23.1.

The main effect is episode restructuring: `OverCapMinutes` falls from 0.00 to 0.00 while `CVaR_0.95(L)` drops from 6.36 to 6.36, but sessions per episode rise from 13.72 to 13.72 and the fraction of returns within 1 / 5 minutes shifts from 0.000 / 0.001 to 0.000 / 0.001.

The cap therefore looks more like long-session truncation than a clean mitigation of total use: the strongest cap point (`T_cap=90`) drives `OverCapMinutes` to 0.00 while still leaving `CumWatch` at 23.1, and `LateNightSessionStartRate` at the default cap moves down from 0.243 to 0.243.
