# A-B-Testing
Project Overview
This project explores the implementation and comparison of two popular multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling. Multi-armed bandit problems involve a trade-off between exploration (trying different arms) and exploitation (choosing the best-known arm). These algorithms are commonly used in scenarios like online advertising, recommendation systems, and more.
Structure of the Project
The project consists of the following files:
1.	bandit.py: This file contains the implementation of the Epsilon Greedy and Thompson Sampling bandit algorithms. Both classes provide methods for pulling arms, updating estimates, and conducting experiments. Additionally, they offer plotting and reporting methods to visualize the learning process and generate a summary report of the experiment.
2.	report.ipynb: This Jupyter notebook demonstrates how to use the bandit algorithms implemented in the bandit.py file. It includes examples of running experiments with different parameters, plotting the results, and generating reports.
Usage
To utilize the bandit algorithms:
1.	Clone or download the repository to your local machine.
2.	Run the logs.py file to set up logging functionalities.
3.	Run the bandit.py file to enable the necessary functionalities for the experiments.
4.	Open the report.ipynb notebook using Jupyter Notebook or JupyterLab.
5.	Execute the notebook cells to run experiments with Epsilon Greedy and Thompson Sampling algorithms, visualize the results, and generate reports.
Saving Experiment Output
Both bandit algorithms save the output of the experiments in CSV files for future reference. The output includes details such as the chosen bandit for each trial, the rewards obtained, and the cumulative rewards and regrets over time.
This repository is designed to be a practical resource for individuals interested in implementing and experimenting with multi-armed bandit algorithms. Whether you're a beginner exploring the concepts or an experienced practitioner looking for a reliable implementation, this project offers a solid foundation. You're encouraged to delve into the code, experiment with different parameters, and adapt it to suit your specific needs and projects.
