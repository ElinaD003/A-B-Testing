from abc import ABC, abstractmethod
import logging
from logs import *
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig
app_logger = logging.getLogger("MAB Application")

# Setting up logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter())
app_logger.addHandler(console_handler)

class Bandit(ABC):
    def __init__(self, true_win_rate):
        self.true_win_rate = true_win_rate
        self.win_rate_estimate = 0
        self.pull_count = 0
        self.regret_estimate = 0

    def __repr__(self):
        return f'An Arm with {self.true_win_rate} Win Rate'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    def report(self, num_trials, results, algorithm="Epsilon Greedy"):
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_avg, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward, suboptimal_count = results
        else:
            cumulative_reward_avg, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward = results

        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_results.csv', index=False)

        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.win_rate_estimate for p in bandits],
            'Algorithm': algorithm
        })

        data_df1.to_csv(f'{algorithm}.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].true_win_rate} - Pulled {bandits[b].pull_count} times - Estimated average reward - {round(bandits[b].win_rate_estimate, 4)} - Estimated average regret - {round(bandits[b].regret_estimate, 4)}')
            print("--------------------------------------------------")

        print(f"Cumulative Reward : {sum(reward)}")
        print(" ")
        print(f"Cumulative Regret : {cumulative_regret[-1]}")
        print(" ")

        if algorithm == 'EpsilonGreedy':
            print(f"Percent suboptimal : {round((float(suboptimal_count) / num_trials), 4)}")

class Visualization:
    def plot1(self, num_trials, results, algorithm='EpsilonGreedy'):
        cumulative_reward_avg = results[0]
        bandits = results[3]

        plt.plot(cumulative_reward_avg, label='Cumulative Average Reward')
        plt.plot(np.ones(num_trials) * max([b.true_win_rate for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        plt.plot(cumulative_reward_avg, label='Cumulative Average Reward')
        plt.plot(np.ones(num_trials) * max([b.true_win_rate for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_eg, results_ts):
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    def __init__(self, true_win_rate):
        super().__init__(true_win_rate)

    def pull(self):
        return np.random.randn() + self.true_win_rate

    def update(self, x):
        self.pull_count += 1.
        self.win_rate_estimate = (1 - 1.0/self.pull_count) * self.win_rate_estimate + 1.0/ self.pull_count * x
        self.regret_estimate = self.true_win_rate - self.win_rate_estimate

    def experiment(self, bandit_win_rates, num_trials, t=1):
        bandits = [EpsilonGreedy(p) for p in bandit_win_rates]
        means = np.array(bandit_win_rates)
        true_best = np.argmax(means)
        suboptimal_count = 0
        EPS = 1/t

        reward = np.empty(num_trials)
        chosen_bandit = np.empty(num_trials)

        for i in range(num_trials):
            p = np.random.random()

            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.win_rate_estimate for b in bandits])

            x = bandits[j].pull()

            bandits[j].update(x)

            if j != true_best:
                suboptimal_count += 1

            reward[i] = x
            chosen_bandit[i] = j

            t += 1
            EPS = 1/t

        cumulative_reward_avg = np.cumsum(reward) / (np.arange(num_trials) + 1)
        cumulative_reward = np.cumsum(reward)

        cumulative_regret = np.empty(num_trials)
        for i in range(len(reward)):
            cumulative_regret[i] = num_trials * max(means) - cumulative_reward[i]

        return cumulative_reward_avg, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, suboptimal_count

class ThompsonSampling(Bandit):
    def __init__(self, true_win_rate):
        super().__init__(true_win_rate)
        self.lambda_ = 1
        self.tau = 1

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_win_rate

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.win_rate_estimate

    def update(self, x):
        self.win_rate_estimate = (self.tau * x + self.lambda_ * self.win_rate_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.pull_count += 1
        self.regret_estimate = self.true_win_rate - self.win_rate_estimate

    def     plot(self, bandits, trial):
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.win_rate_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.true_win_rate:.4f}, num plays: {b.pull_count}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, bandit_win_rates, num_trials):
        bandits = [ThompsonSampling(m) for m in bandit_win_rates]

        sample_points = [5, 20, 50, 100, 200, 500, 1000, 1999, 5000, 10000, 19999]
        reward = np.empty(num_trials)
        chosen_bandit = np.empty(num_trials)

        for i in range(num_trials):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()

            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_avg = np.cumsum(reward) / (np.arange(num_trials) + 1)
        cumulative_reward = np.cumsum(reward)

        cumulative_regret = np.empty(num_trials)

        for i in range(len(reward)):
            cumulative_regret[i] = num_trials * max([b.true_win_rate for b in bandits]) - cumulative_reward[i]

        return cumulative_reward_avg, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward

def comparison(num_trials, results_eg, results_ts):
    cumulative_reward_avg_eg = results_eg[0]
    cumulative_reward_avg_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    print(f"Total Reward Epsilon Greedy : {sum(reward_eg)}")
    print(f"Total Reward Thompson Sampling : {sum(reward_ts)}")
    print(" ")
    print(f"Total Regret Epsilon Greedy : {regret_eg}")
    print(f"Total Regret Thompson Sampling : {regret_ts}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_avg_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(num_trials) * max([b.true_win_rate for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_avg_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(num_trials) * max([b.true_win_rate for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")

    plt.tight_layout()
    plt.show()

bandit_win_rates = [1, 2, 3, 4]

num_trials = 20000

epsilon_greedy = EpsilonGreedy(0.5)
thompson_sampling = ThompsonSampling(0.5)

results_eg = epsilon_greedy.experiment(bandit_win_rates, num_trials)
results_ts = thompson_sampling.experiment(bandit_win_rates, num_trials)

visualization = Visualization()
visualization.plot1(num_trials, results_eg)
visualization.plot2(results_eg, results_ts)

comparison(num_trials, results_eg, results_ts)

