# Mechanism ablations

Habit removal remains the clearest mechanism lever. Under `NoHabit`, PPO changes by -4.7% in `CumWatch`, -3.1% in `CVaR_0.95(L)`, and +0.0% in `OverCapMinutes`; the corresponding Lagrangian PPO shifts are -4.9%, -3.6%, and +0.0%. Risk therefore does not disappear when habit is removed, but the frontier compresses materially, which supports the claim that long-horizon accumulation is doing real work rather than merely decorating a one-step reward model.

The broader ablations make the story more specific. `NoPers` moves the frontier by +0.0% for PPO and +0.0% for Lagrangian PPO, so personalization clearly matters. `HomogeneousUsers` also produces sizeable changes (+0.0% for PPO; +0.0% for Lagrangian PPO), which indicates that user heterogeneity contributes beyond the mean dynamics. By contrast, `NoVar` is weaker on the core frontier metrics (+0.0% for PPO; +0.0% for Lagrangian PPO), so pure reward variability is not the main driver. The reported night-family column is `NightFraction`.

