# HW2 – A/B Testing: Multi-Armed Bandits

## Overview

This homework implements and compares two bandit algorithms:

- Epsilon-Greedy
- Thompson Sampling

Both algorithms are used to select among four advertising options (bandits) with different expected rewards.

## Experiment Settings

- Number of bandits: 4
- Expected rewards: `[1, 2, 3, 4]`
- Number of trials: 20,000 per algorithm
- Reward distribution: Gaussian (Normal)
- Reward precision: 1.0

The Gaussian reward distribution and precision are modeling assumptions used for this implementation.

## Algorithms

### 1. Epsilon-Greedy

The Epsilon-Greedy algorithm balances exploration and exploitation. The exploration probability decreases over time according to:

\[
\epsilon(t)=\frac{1}{t}
\]

### 2. Thompson Sampling

Thompson Sampling uses a Bayesian approach to select bandits. It samples from the posterior reward distributions and selects the bandit with the highest sampled value.

The implementation assumes known reward precision and uses a Gaussian prior.

## Visualizations

The implementation generates:

- Learning-process graphs for both algorithms, using linear and logarithmic scales.
- A cumulative reward comparison.
- A cumulative regret comparison.

## Results

The experiment records the selected bandit, observed reward, and algorithm in a CSV file.

The output file is `rewards.csv`, containing the following columns:

- `Bandit`
- `Reward`
- `Algorithm`

Each algorithm contributes 20,000 records.

## Installation and Execution

Install the required dependencies:

```bash
pip install -r HW2/requirements.txt
```

Run the experiment from the repository root:

```bash
python HW2/Bandit.py
```

The script runs both algorithms, prints their cumulative rewards and regrets, generates the visualizations, and saves the results to `HW2/rewards.csv`.

## Files

- `Bandit.py` – Algorithm implementations and visualizations.
- `requirements.txt` – Required Python packages.
- `rewards.csv` – Experiment results generated when the script runs.
