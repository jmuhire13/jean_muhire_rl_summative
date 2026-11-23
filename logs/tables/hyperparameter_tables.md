## DQN

| Run | Change from baseline | Eval reward (mean ± std) | Graduations held | Relapses | In crisis at end | Lasting score |
|---|---|---|---|---|---|---|
| dqn_01_baseline | baseline | -64.3 ± 36.1 | 4.40 | 0.40 | 4.70 | 0.39 |
| dqn_02_lr_low | learning_rate=0.0001 | -83.2 ± 23.5 | 0.00 | 0.00 | 11.75 | 0.18 |
| dqn_03_lr_high | learning_rate=0.005 | -115.9 ± 27.2 | 0.00 | 3.35 | 11.50 | 0.18 |
| dqn_04_gamma_090 | gamma=0.9 | -87.1 ± 26.3 | 0.75 | 0.15 | 3.00 | 0.38 |
| dqn_05_gamma_0999 | gamma=0.999 | -121.7 ± 27.3 | 0.70 | 4.10 | 6.85 | 0.29 |
| dqn_06_batch_32 | batch_size=32 | -67.8 ± 30.0 | 0.00 | 0.00 | 9.55 | 0.18 |
| dqn_07_small_buffer | buffer_size=5000 | -74.8 ± 41.6 | 3.65 | 1.55 | 4.50 | 0.39 |
| dqn_08_eps_end_001 | eps_end=0.01 | -106.7 ± 27.7 | 0.00 | 6.30 | 10.90 | 0.19 |
| dqn_09_eps_end_020 | eps_end=0.2 | -108.4 ± 34.4 | 1.35 | 4.15 | 6.35 | 0.32 |
| dqn_10_fast_decay | eps_decay_fraction=0.1 | -107.7 ± 18.9 | 0.00 | 0.50 | 11.80 | 0.17 |
| dqn_11_slow_decay | eps_decay_fraction=0.9 | -98.4 ± 28.3 | 1.05 | 1.95 | 6.15 | 0.31 |
| dqn_12_target_freq_50 | target_update_interval=50 | -88.7 ± 40.5 | 3.45 | 1.00 | 3.10 | 0.42 |
| dqn_13_soft_target | target_update=soft, target_tau=0.01 | -84.0 ± 20.6 | 0.00 | 3.30 | 9.10 | 0.18 |
| dqn_14_double_dqn | double_dqn=True | -117.6 ± 20.0 | 0.00 | 2.15 | 11.90 | 0.17 |
| dqn_15_small_network | hidden_layers=[64] | -53.4 ± 23.3 | 0.00 | 0.90 | 8.10 | 0.19 |
| dqn_16_l2_loss | loss_function=l2 | -80.1 ± 19.6 | 0.00 | 1.35 | 11.45 | 0.17 |
| dqn_17_reward_scale | reward_scale=0.1 | -21.4 ± 40.3 | 5.80 | 0.15 | 0.80 | 0.48 |

## REINFORCE

| Run | Change from baseline | Eval reward (mean ± std) | Graduations held | Relapses | In crisis at end | Lasting score |
|---|---|---|---|---|---|---|
| reinforce_01_baseline | baseline | -146.9 ± 15.0 | 0.00 | 0.00 | 11.90 | 0.17 |
| reinforce_02_lr_low | learning_rate=0.0001 | -65.2 ± 30.4 | 0.00 | 0.00 | 1.20 | 0.23 |
| reinforce_03_lr_high | learning_rate=0.005 | -143.6 ± 14.7 | 0.00 | 0.00 | 11.90 | 0.17 |
| reinforce_04_gamma_090 | gamma=0.9 | -105.7 ± 14.9 | 0.00 | 0.00 | 11.95 | 0.17 |
| reinforce_05_gamma_0999 | gamma=0.999 | -8.7 ± 22.4 | 0.00 | 0.00 | 0.00 | 0.45 |
| reinforce_06_no_baseline | use_baseline=False | -120.3 ± 16.4 | 0.00 | 0.00 | 11.95 | 0.17 |
| reinforce_07_no_normalize | normalize_returns=False | -123.4 ± 19.1 | 0.00 | 0.00 | 11.90 | 0.17 |
| reinforce_08_no_entropy | entropy_coeff=0.0 | -112.4 ± 13.0 | 0.00 | 0.00 | 11.90 | 0.17 |
| reinforce_09_high_entropy | entropy_coeff=0.05 | -6.0 ± 20.2 | 0.00 | 0.00 | 0.00 | 0.46 |
| reinforce_10_batch_1 | episodes_per_batch=1 | -112.8 ± 17.5 | 0.00 | 0.00 | 11.90 | 0.17 |
| reinforce_11_batch_20 | episodes_per_batch=20 | -8.7 ± 22.4 | 0.00 | 0.00 | 0.00 | 0.45 |
| reinforce_12_small_network | hidden_layers=[64] | -8.7 ± 22.4 | 0.00 | 0.00 | 0.00 | 0.45 |

## PPO

| Run | Change from baseline | Eval reward (mean ± std) | Graduations held | Relapses | In crisis at end | Lasting score |
|---|---|---|---|---|---|---|
| ppo_01_baseline | baseline | -38.0 ± 16.0 | 0.00 | 0.00 | 2.40 | 0.22 |
| ppo_02_lr_low | learning_rate=0.0001 | -125.2 ± 18.0 | 0.00 | 0.00 | 11.75 | 0.17 |
| ppo_03_lr_high | learning_rate=0.001 | -33.0 ± 18.4 | 0.00 | 0.00 | 1.25 | 0.23 |
| ppo_04_clip_010 | clip_epsilon=0.1 | -32.9 ± 17.9 | 0.00 | 0.00 | 2.15 | 0.22 |
| ppo_05_clip_030 | clip_epsilon=0.3 | -36.1 ± 17.1 | 0.00 | 0.00 | 2.30 | 0.22 |
| ppo_06_epochs_10 | epochs=10 | -37.8 ± 16.5 | 0.00 | 0.00 | 1.90 | 0.22 |
| ppo_07_rollout_short | rollout_steps=520 | -39.0 ± 20.5 | 0.00 | 0.00 | 5.30 | 0.20 |
| ppo_08_rollout_long | rollout_steps=2080 | -40.3 ± 17.0 | 0.00 | 0.00 | 1.95 | 0.22 |
| ppo_09_lambda_080 | gae_lambda=0.8 | -35.8 ± 19.7 | 0.00 | 0.00 | 1.70 | 0.22 |
| ppo_10_lambda_100 | gae_lambda=1.0 | -8.7 ± 22.4 | 0.00 | 0.00 | 0.00 | 0.45 |
| ppo_11_no_entropy | entropy_coeff=0.0 | -41.3 ± 17.1 | 0.00 | 0.00 | 1.90 | 0.22 |
| ppo_12_high_entropy | entropy_coeff=0.05 | -35.4 ± 22.8 | 0.00 | 0.00 | 4.75 | 0.20 |
| ppo_13_gamma_090 | gamma=0.9 | -105.1 ± 15.9 | 0.00 | 0.00 | 11.90 | 0.17 |

## A2C

| Run | Change from baseline | Eval reward (mean ± std) | Graduations held | Relapses | In crisis at end | Lasting score |
|---|---|---|---|---|---|---|
| a2c_01_baseline | baseline | -33.1 ± 18.2 | 0.00 | 0.00 | 2.20 | 0.22 |
| a2c_02_lr_low | learning_rate=0.0001 | -120.7 ± 12.8 | 0.00 | 0.00 | 11.90 | 0.17 |
| a2c_03_lr_high | learning_rate=0.003 | -146.9 ± 15.0 | 0.00 | 0.00 | 11.90 | 0.17 |
| a2c_04_steps_8 | rollout_steps=8 | -35.7 ± 20.0 | 0.00 | 0.00 | 3.25 | 0.21 |
| a2c_05_steps_128 | rollout_steps=128 | -138.0 ± 14.9 | 0.00 | 0.00 | 11.90 | 0.17 |
| a2c_06_lambda_095 | gae_lambda=0.95 | -36.3 ± 17.0 | 0.00 | 0.00 | 1.60 | 0.22 |
| a2c_07_no_entropy | entropy_coeff=0.0 | -34.6 ± 19.2 | 0.00 | 0.00 | 3.80 | 0.21 |
| a2c_08_high_entropy | entropy_coeff=0.05 | -40.2 ± 23.8 | 0.00 | 0.00 | 5.90 | 0.20 |
| a2c_09_critic_10 | critic_coeff=1.0 | -34.8 ± 17.7 | 0.00 | 0.00 | 2.40 | 0.21 |
| a2c_10_gamma_090 | gamma=0.9 | -104.2 ± 16.0 | 0.00 | 0.00 | 11.90 | 0.17 |
| a2c_11_small_network | hidden_layers=[64] | -146.9 ± 15.0 | 0.00 | 0.00 | 11.90 | 0.17 |

