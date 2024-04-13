# A-B-Testing
# Structure of the Project

The project consists of the following files:

bandit.py: This file contains the implementation of the Epsilon Greedy and Thompson Sampling bandit algorithms. Both classes provide methods for pulling arms, updating estimates, and conducting experiments. Additionally, they offer plotting and reporting methods to visualize the learning process and generate a summary report of the experiment.

report.ipynb: This notebook demonstrates how to use the bandit algorithms implemented in the bandit.py file. It includes examples of running experiments with different parameters, plotting the results, and generating reports.

logs.py: This file sets up logging functionalities for the project.

# Usage

To utilize the bandit algorithms:

Clone or download the repository.

Run the logs.py file to set up logging functionalities.

Run the bandit.py file to enable the necessary functionalities for the experiments.

Open the report.ipynb notebook.

Execute the notebook cells to run experiments with Epsilon Greedy and Thompson Sampling algorithms, visualize the results, and generate reports.

# Saving Experiment Output

Algorithms save the output of the experiments in CSV files for future reference. The output includes details such as the chosen bandit for each trial, the rewards obtained, and the cumulative rewards and regrets over time. They provide a structured format for easy analysis and retrievel of experimental results.


# About

This repository is designed to be a practical resource for individuals interested in implementing and experimenting with multi-armed bandit algorithms. Whether you're a beginner exploring the concepts or an experienced practitioner looking for a reliable implementation, this project offers a solid foundation. You're encouraged to delve into the code, experiment with different parameters, and adapt it to suit your specific needs and projects.
