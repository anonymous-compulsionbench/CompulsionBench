# Mechanism ablations

Habit removal remains the clearest mechanism lever. Under `NoHabit`, PPO changes by -7.5% in `CumWatch`, -5.4% in `CVaR_0.95(L)`, and -24.7% in `OverCapMinutes`; the corresponding Lagrangian PPO shifts are -4.2%, -2.2%, and -11.0%. Risk therefore does not disappear when habit is removed, but the frontier compresses materially, which supports the claim that long-horizon accumulation is doing real work rather than merely decorating a one-step reward model.

The broader ablations make the story more specific. `NoPers` moves the frontier by -0.8% for PPO and -0.1% for Lagrangian PPO. `HomogeneousUsers` produces sizeable changes (-21.1% for PPO; -18.9% for Lagrangian PPO), which indicates that user heterogeneity contributes beyond the mean dynamics. By contrast, `NoVar` shows sizeable effects on the core frontier metrics (-23.4% for PPO; +0.0% for Lagrangian PPO). The reported night-family column is `LateNightSessionStartRate`.

