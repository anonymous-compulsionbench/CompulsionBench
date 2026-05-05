# Proxy Orthogonality Audit

- promotion abs(policy-mean corr) threshold: 0.95
- promoted main-text night proxy: NightFraction
- main-text scorecard risk metrics: OverCapMinutes, NightFraction
- main-text constrained channels: OverCapMinutes

Promoting NightFraction to the main-text scorecard because its |policy-mean corr| with CumWatch is 0.410 and it changes at least one policy comparison.

## Candidate metrics
- NightMinutes: policy-mean corr=0.995, episode corr=0.704, changed_pair_count=1, eligible_main_text=False
  examples: LeastRecentPolicy vs NoveltyGreedyPolicy
- NightFraction: policy-mean corr=0.410, episode corr=0.097, changed_pair_count=8, eligible_main_text=True
  examples: Myopic vs PPO+AutoplayOff, Myopic vs Lagrangian PPO, RoundRobinPolicy vs LeastRecentPolicy, LeastRecentPolicy vs NoveltyGreedyPolicy, NoveltyGreedyPolicy vs PPO+AutoplayOff
- LateNightSessionStartRate: policy-mean corr=0.679, episode corr=0.085, changed_pair_count=6, eligible_main_text=True
  examples: Myopic vs PPO+AutoplayOff, RoundRobinPolicy vs LeastRecentPolicy, LeastRecentPolicy vs NoveltyGreedyPolicy, PPO vs PPO+AutoplayOff, PPO+AutoplayOff vs PPO + SessionCap(120)
