# Deep Q-Networks (DQN) from Scratch — PyTorch + Gymnasium

An educational implementation of **Deep Q-Networks** (Mnih et al., 2015) built
on plain PyTorch: a neural network that learns to act from nothing but scalar
rewards. The **replay buffer**, **target network**, **epsilon-greedy schedule**,
and the classic upgrades — **Double DQN** and the **dueling architecture** — are
all implemented by hand and toggleable from the command line. Trained on
`CartPole-v1`, it solves the task (100-episode moving average ≥ 475) within a
few minutes on CPU.

## What's Implemented

| File | Contents |
|------|----------|
| `network.py` | `QNetwork` — MLP `Q_θ(s, ·)`; `DuelingQNetwork` — value/advantage streams (Wang et al. 2016); `make_q_network` factory |
| `agent.py` | `ReplayBuffer` — preallocated NumPy ring buffer; `DQNAgent` — linear ε-greedy schedule, Huber loss, Double DQN target, hard target sync, save/load |
| `train.py` | CLI training loop → `runs/history.csv`, `best.pt` / `last.pt` / `args.json`, early stop at the solved threshold |
| `evaluate.py` | Greedy (ε = 0) rollout of a checkpoint → mean/std/min/max reward, optional `--render` window |
| `plots.py` | `rewards.png` — raw rewards + 100-episode moving average + solved-threshold line |
| `tests/` | 11 pytest cases: buffer semantics, network shapes, dueling invariance, ε schedule, sync logic, save/load, 300-step end-to-end |

**Toggles:** `--no-double` disables Double DQN (vanilla `max` target);
`--dueling` switches to the dueling architecture. Double DQN is on by default.

## Background: From Q-Learning to Deep Q-Networks

**Bellman optimality.** For a Markov decision process with discount `γ`, the
optimal action-value function satisfies

```
Q*(s, a) = E[ r + γ · max_a' Q*(s', a') ]
```

i.e. the value of taking `a` in `s` is the immediate reward plus the
discounted value of acting optimally thereafter. **Q-learning** (Watkins, 1989)
turns this into an update rule with the TD target

```
y      = r + γ · max_a' Q(s', a')          (bootstrap off-policy)
Q(s,a) ← Q(s,a) + α · (y − Q(s,a))
```

**Why a neural network — and why that breaks Q-learning.** Tabular Q-learning
needs one entry per state-action pair; CartPole's state space is continuous, so
we approximate `Q*(s, a) ≈ Q_θ(s, a)` with an MLP and minimize the squared
(here Huber) TD error `(y − Q_θ(s,a))²` by SGD. Two instabilities appear:

1. **Moving target.** `y` is computed with the *same* network being trained —
   like chasing your own shadow. DQN holds a **target network** `θ⁻` (a frozen
   copy, hard-synced every `C` gradient steps) fixed inside the target:
   `y = r + γ·(1 − done)·Q_θ⁻(s', ·)`. Between syncs the regression target is
   stationary, which is what makes training stable.
2. **Correlated samples.** Consecutive transitions are highly correlated
   (`s_{t+1}` barely differs from `s_t`), so naive online SGD violates the
   i.i.d. assumption. The **replay buffer** stores transitions `(s, a, r, s',
   done)` in a large ring buffer and samples minibatches uniformly at random —
   approximately i.i.d. samples, and every transition is reused across many
   gradient steps instead of once.

**Double DQN — fixing overestimation.** The `max` in the target is a
*maximization bias*: with noisy estimates, `max_a Q̂(s', a)` systematically
overestimates `max_a Q(s', a)` (noise inflates the maximum). Since DQN
bootstraps off these optimistic values, errors compound. **Double Q-learning**
(van Hasselt, 2010) decouples *selection* from *evaluation* with two
independent estimators; **Double DQN** (van Hasselt et al., 2016) gets the same
effect almost for free by using the online network for selection and the target
network for evaluation:

```
a*(s') = argmax_a Q_θ(s', a)                       # online picks the action
y      = r + γ·(1 − done)·Q_θ⁻(s', a*(s'))         # target scores it
```

**Dueling architecture.** For many states, the choice of action barely matters
— what matters is the state itself. The dueling network (Wang et al., 2016)
therefore estimates two streams and recombines them:

```
Q(s, a) = V(s) + A(s, a) − mean_a' A(s, a')
```

`V(s)` captures "how good is this state", `A(s, a)` "how much better is `a`
than average here". The mean-subtraction makes the split identifiable
(otherwise `V` and `A` could drift by arbitrary constants together); `V` can
now be learned without visiting every action in every state.

## Quickstart

```bash
pip install -r requirements.txt

# train — CPU, ~5-10 min; expect the 100-ep moving avg to reach 475
# somewhere around episode 300-500
python train.py --env CartPole-v1 --episodes 600

# ablations (compare learning curves!)
python train.py --no-double --out-dir runs_vanilla     # vanilla DQN target
python train.py --dueling   --out-dir runs_dueling     # dueling architecture

# evaluate the best checkpoint greedily
python evaluate.py --checkpoint runs/best.pt --env CartPole-v1 --episodes 20
python evaluate.py --checkpoint runs/best.pt --render  # watch it play

# learning curve
python plots.py --history runs/history.csv             # → rewards.png
```

Artifacts land in `runs/`: `history.csv` (per-episode reward, steps, ε,
100-ep moving average), `best.pt` (saved whenever the moving average improves),
`last.pt`, and `args.json` (exact hyperparameters).

### Optional: LunarLander

`gymnasium[box2d]` is *not* required for this project. To try the harder
Box2D task, install the extra (needs the SWIG system package: `apt install
swig` or `brew install swig`):

```bash
pip install "gymnasium[box2d]"
python train.py --env LunarLander-v3 --episodes 1500 --eps-decay-steps 100_000
```

LunarLander needs substantially more episodes and a slower ε decay than
CartPole; expect ~1000+ episodes before it consistently lands.

## Results (typical run)

Numbers below are what a typical run on CartPole-v1 looks like with these
defaults (`--seed 42`); individual runs vary noticeably — run a couple of seeds
before drawing conclusions:

| Variant | Solves (100-ep avg ≥ 475) | Greedy eval reward | Notes |
|---------|---------------------------|--------------------|-------|
| Double DQN (default) | ~300–500 episodes | ~500 | stable, our default |
| Vanilla DQN (`--no-double`) | ~400–600 episodes | ~500 | more run-to-run variance |
| Dueling (`--dueling`) | ~350–550 episodes | ~500 | similar to default on CartPole |

CartPole is "solved" at a 100-episode moving average of 475 (episodes are
capped at 500 reward); `train.py` early-stops at that threshold using
`env.spec.reward_threshold`.

**Suggested ablation.** Run the default and `--no-double` with 2–3 seeds each
and compare the learning curves in `rewards.png`: the Double DQN target's lower
overestimation bias typically shows up as *lower variance across seeds* and
fewer collapse-and-recover dips than in the vanilla curve.

## Tests

```bash
python -m pytest tests/ -v
```

11 offline, CPU-only cases: buffer shapes/dtypes and FIFO overwrite; `done`
flag storage; Q-network output shapes and the `make_q_network` factory; the
dueling `A − mean(A)` invariance (shifting all advantages leaves Q unchanged);
action validity and greedy determinism; exact linear ε schedule (start, middle,
end, clamp); one `train_step` on a synthetic batch (finite loss, online moves
off the target); explicit and automatic target-network sync; save/load
round-trip (including the architecture flags stored in the checkpoint); and a
300-step end-to-end CartPole agent loop with `learning_starts=100`.

## Topics Covered By Courses (for further study)

- UC Berkeley **CS188** — MDPs, Bellman equations, tabular Q-learning
- UC Berkeley **CS285** — deep RL: value-function approximation, target
  networks, replay buffers, off-policy corrections
- Stanford **CS234** — TD learning, DQN and its variants (Double / Dueling /
  Prioritized Replay / Rainbow)
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  ch. 6 (TD learning) and ch. 16 (applications to games)

## References

1. Mnih, V. et al. *Playing Atari with Deep Reinforcement Learning*,
   arXiv:1312.5602 (NeurIPS Deep Learning Workshop, 2013)
2. Mnih, V. et al. *Human-level control through deep reinforcement learning*,
   Nature 518, 529–533 (2015) — the DQN paper
3. van Hasselt, H. *Double Q-learning*, NeurIPS 2010
4. van Hasselt, H., Guez, A. & Silver, D. *Deep Reinforcement Learning with
   Double Q-learning*, AAAI 2016, arXiv:1509.06461
5. Wang, Z. et al. *Dueling Network Architectures for Deep Reinforcement
   Learning*, ICML 2016, arXiv:1511.06581
6. Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*,
   2nd ed., MIT Press (2018)
