"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from pathlib import Path

# Experiment settings
BANDIT_REWARDS = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
REWARD_PRECISION = 1.0



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#



class Visualization:
    """Visualization of the bandit experiments.

    Parameters
    ----------
    epsilon_greedy : EpsilonGreedy
        Epsilon-Greedy bandit after the experiment has been run.
    thompson_sampling : ThompsonSampling
        Thompson Sampling bandit after the experiment has been run.
    """

    def __init__(self, epsilon_greedy, thompson_sampling):
        """Store the two algorithms that will be visualized.

        Parameters
        ----------
        epsilon_greedy : EpsilonGreedy
            Epsilon-Greedy bandit after the experiment has been run.
        thompson_sampling : ThompsonSampling
            Thompson Sampling bandit after the experiment has been run.
        """
        self.epsilon_greedy = epsilon_greedy
        self.thompson_sampling = thompson_sampling

    def plot1(self):
        """Visualize the learning process of each algorithm.

        For each algorithm, plots the estimated average reward of every
        bandit over the trials, using a linear and a logarithmic x-axis.

        Returns
        -------
        None
        """
        algorithms = [
            self.epsilon_greedy,
            self.thompson_sampling
        ]

        for algorithm in algorithms:

            history = np.array(algorithm.estimate_history)
            trials = np.arange(1, len(history) + 1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Linear scale
            for arm in range(len(BANDIT_REWARDS)):
                axes[0].plot(
                    trials,
                    history[:, arm],
                    label=f"Bandit {arm + 1}"
                )

            axes[0].set_title(
                f"{algorithm} - Linear Scale"
            )
            axes[0].set_xlabel("Number of Trials")
            axes[0].set_ylabel("Estimated Average Reward")
            axes[0].legend()
            axes[0].grid(True)

            # Logarithmic scale
            for arm in range(len(BANDIT_REWARDS)):
                axes[1].plot(
                    trials,
                    history[:, arm],
                    label=f"Bandit {arm + 1}"
                )

            axes[1].set_xscale("log")
            axes[1].set_title(
                f"{algorithm} - Logarithmic Scale"
            )
            axes[1].set_xlabel("Number of Trials (log scale)")
            axes[1].set_ylabel("Estimated Average Reward")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    def plot2(self):
        """Compare Epsilon-Greedy and Thompson Sampling.

        Left plot: cumulative rewards of both algorithms.
        Right plot: cumulative regrets of both algorithms.

        Returns
        -------
        None
        """

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Compare cumulative rewards
        axes[0].plot(
            self.epsilon_greedy.cumulative_rewards,
            label="Epsilon-Greedy"
        )

        axes[0].plot(
            self.thompson_sampling.cumulative_rewards,
            label="Thompson Sampling"
        )

        axes[0].set_title("Cumulative Rewards Comparison")
        axes[0].set_xlabel("Number of Trials")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()
        axes[0].grid(True)

        # Compare cumulative regrets
        axes[1].plot(
            self.epsilon_greedy.cumulative_regrets,
            label="Epsilon-Greedy"
        )

        axes[1].plot(
            self.thompson_sampling.cumulative_regrets,
            label="Thompson Sampling"
        )

        axes[1].set_title("Cumulative Regret Comparison")
        axes[1].set_xlabel("Number of Trials")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy bandit algorithm with epsilon decaying as 1/t.

    Parameters
    ----------
    p : list of float
        True mean reward of each bandit.
    """

    def __init__(self, p):
        """Initialize the algorithm state.

        Parameters
        ----------
        p : list of float
            True mean reward of each bandit.
        """
        self.p = np.array(p, dtype=float)

        self.counts = np.zeros(len(p), dtype=int)
        self.estimates = np.zeros(len(p))
        self.estimate_history = []

        self.rewards = []
        self.selected_bandits = []
        self.regrets = []

        self.cumulative_rewards = []
        self.cumulative_regrets = []

        self.t = 0
        self.last_arm = None
        self.last_reward = None

    def __repr__(self):
        """Return the name of the algorithm.

        Returns
        -------
        str
            Algorithm name.
        """
        return "EpsilonGreedy Algorithm"

    def pull(self):
        """Select a bandit and generate its reward.

        With probability epsilon = 1/t a random bandit is explored,
        otherwise the bandit with the highest estimate is exploited.
        The reward is drawn from a Gaussian distribution.

        Returns
        -------
        float
            Observed reward of the selected bandit.
        """
        self.t += 1

        # Epsilon decay: epsilon = 1/t
        epsilon = 1 / self.t

        # Exploration or exploitation
        if random.random() < epsilon:
            arm = random.randrange(len(self.p))
        else:
            arm = np.argmax(self.estimates)

        # Generate reward using a Gaussian distribution
        reward = np.random.normal(
            self.p[arm],
            1 / np.sqrt(REWARD_PRECISION)
        )

        self.last_arm = arm
        self.last_reward = reward

        return reward

    def update(self):
        """Update estimates, rewards, regrets and cumulative values.

        Uses the last pulled bandit and its reward to update the
        running average estimate and the stored statistics.

        Returns
        -------
        None
        """
        arm = self.last_arm
        reward = self.last_reward

        # Update the number of selections
        self.counts[arm] += 1

        # Update estimated average reward
        self.estimates[arm] += (
            reward - self.estimates[arm]
        ) / self.counts[arm]

        self.rewards.append(reward)
        self.selected_bandits.append(arm + 1)

        # Calculate regret
        regret = max(self.p) - self.p[arm]
        self.regrets.append(regret)

        # Calculate cumulative reward and regret
        previous_reward = self.cumulative_rewards[-1] if self.cumulative_rewards else 0
        previous_regret = self.cumulative_regrets[-1] if self.cumulative_regrets else 0

        self.cumulative_rewards.append(previous_reward + reward)
        self.cumulative_regrets.append(previous_regret + regret)
        
        self.estimate_history.append(self.estimates.copy())

    def experiment(self):
        """Run the experiment for NUMBER_OF_TRIALS trials.

        Returns
        -------
        None
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.pull()
            self.update()

        logger.info(
            f"Epsilon-Greedy completed {NUMBER_OF_TRIALS} trials."
        )

    def report(self):
        """Store the results in a CSV file and log the summary.

        Appends {Bandit, Reward, Algorithm} rows to ``rewards.csv`` and
        logs the cumulative and average reward and regret.

        Returns
        -------
        None
        """
        csv_path = Path(__file__).resolve().parent / "rewards.csv"

        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Bandit", "Reward", "Algorithm"])

            for bandit, reward in zip(
                self.selected_bandits, self.rewards
            ):
                writer.writerow([
                    bandit,
                    reward,
                    "Epsilon-Greedy"
                ])

        logger.info(
            f"Epsilon-Greedy cumulative reward: "
            f"{sum(self.rewards):.2f}"
        )

        logger.info(
            f"Epsilon-Greedy cumulative regret: "
            f"{sum(self.regrets):.2f}"
        )

        logger.info(
            f"Epsilon-Greedy average reward: "
            f"{np.mean(self.rewards):.4f}"
        )

        logger.info(
            f"Epsilon-Greedy average regret: "
            f"{np.mean(self.regrets):.4f}"
        )


#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling bandit algorithm with known precision.

    Uses a Gaussian prior and a Gaussian likelihood with known
    reward precision.

    Parameters
    ----------
    p : list of float
        True mean reward of each bandit.
    """

    def __init__(self, p):
        """Initialize the prior and the algorithm state.

        Parameters
        ----------
        p : list of float
            True mean reward of each bandit.
        """
        self.p = np.array(p, dtype=float)

        # Prior parameters
        self.prior_mean = 0.0
        self.prior_precision = 1.0

        # Observed data
        self.counts = np.zeros(len(p), dtype=int)
        self.reward_sums = np.zeros(len(p))
        self.estimate_history = []

        self.rewards = []
        self.selected_bandits = []
        self.regrets = []

        self.cumulative_rewards = []
        self.cumulative_regrets = []

        self.last_arm = None
        self.last_reward = None

    def __repr__(self):
        """Return the name of the algorithm.

        Returns
        -------
        str
            Algorithm name.
        """
        return "Thompson Sampling Algorithm"

    def pull(self):
        """Select a bandit by sampling from the posteriors.

        A value is sampled from the posterior of every bandit and the
        bandit with the highest sample is chosen. The reward is drawn
        from a Gaussian distribution.

        Returns
        -------
        float
            Observed reward of the selected bandit.
        """
        sampled_values = []

        for arm in range(len(self.p)):

            # Calculate posterior precision
            posterior_precision = (
                self.prior_precision
                + self.counts[arm] * REWARD_PRECISION
            )

            # Calculate posterior mean
            posterior_mean = (
                self.prior_mean * self.prior_precision
                + self.reward_sums[arm] * REWARD_PRECISION
            ) / posterior_precision

            # Sample from posterior distribution
            sampled_value = np.random.normal(
                posterior_mean,
                1 / np.sqrt(posterior_precision)
            )

            sampled_values.append(sampled_value)

        # Select the arm with the highest sampled value
        arm = int(np.argmax(sampled_values))

        # Generate the observed reward
        reward = np.random.normal(
            self.p[arm],
            1 / np.sqrt(REWARD_PRECISION)
        )

        self.last_arm = arm
        self.last_reward = reward

        return reward

    def update(self):
        """Update posteriors, rewards, regrets and cumulative values.

        Returns
        -------
        None
        """
        arm = self.last_arm
        reward = self.last_reward

        # Update observations
        self.counts[arm] += 1
        self.reward_sums[arm] += reward

        posterior_means = (
            self.prior_mean * self.prior_precision
            + self.reward_sums * REWARD_PRECISION
        ) / (
            self.prior_precision
            + self.counts * REWARD_PRECISION
        )

        self.estimate_history.append(posterior_means.copy())

        self.rewards.append(reward)
        self.selected_bandits.append(arm + 1)

        # Calculate regret
        regret = max(self.p) - self.p[arm]
        self.regrets.append(regret)

        # Calculate cumulative reward and regret
        previous_reward = self.cumulative_rewards[-1] if self.cumulative_rewards else 0
        previous_regret = self.cumulative_regrets[-1] if self.cumulative_regrets else 0

        self.cumulative_rewards.append(previous_reward + reward)
        self.cumulative_regrets.append(previous_regret + regret)

    def experiment(self):
        """Run the experiment for NUMBER_OF_TRIALS trials.

        Returns
        -------
        None
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.pull()
            self.update()

        logger.info(
            f"Thompson Sampling completed "
            f"{NUMBER_OF_TRIALS} trials."
        )

    def report(self):
        """Store the results in a CSV file and log the summary.

        Appends {Bandit, Reward, Algorithm} rows to ``rewards.csv`` and
        logs the cumulative and average reward and regret.

        Returns
        -------
        None
        """
        csv_path = Path(__file__).resolve().parent / "rewards.csv"

        file_exists = (
            csv_path.exists() and csv_path.stat().st_size > 0
        )

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Bandit", "Reward", "Algorithm"])

            for bandit, reward in zip(
                self.selected_bandits, self.rewards
            ):
                writer.writerow([
                    bandit,
                    reward,
                    "Thompson Sampling"
                ])

        logger.info(
            f"Thompson Sampling cumulative reward: "
            f"{sum(self.rewards):.2f}"
        )

        logger.info(
            f"Thompson Sampling cumulative regret: "
            f"{sum(self.regrets):.2f}"
        )

        logger.info(
            f"Thompson Sampling average reward: "
            f"{np.mean(self.rewards):.4f}"
        )

        logger.info(
            f"Thompson Sampling average regret: "
            f"{np.mean(self.regrets):.4f}"
        )






def comparison():
    """Run both algorithms and compare them.

    Runs Epsilon-Greedy and Thompson Sampling, saves the results to
    ``rewards.csv``, and generates the visual comparisons.

    Returns
    -------
    None
    """

    logger.info("Starting A/B testing experiment...")

    # Remove previous results to create a clean CSV
    csv_path = Path(__file__).resolve().parent / "rewards.csv"

    if csv_path.exists():
        csv_path.unlink()

    # Create both algorithms
    epsilon_greedy = EpsilonGreedy(BANDIT_REWARDS)
    thompson_sampling = ThompsonSampling(BANDIT_REWARDS)

    # Run experiments
    epsilon_greedy.experiment()
    thompson_sampling.experiment()

    # Generate reports and save CSV
    epsilon_greedy.report()
    thompson_sampling.report()

    # Compare both algorithms visually
    visualization = Visualization(
        epsilon_greedy,
        thompson_sampling
    )

    visualization.plot1()
    visualization.plot2()

    logger.info("A/B testing experiment completed successfully.")


if __name__ == '__main__':

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()