# The Resistance Agent

This agent was done for the Game AI course of University of Essex.

These files should be in conjunction with [the resistance framework](https://github.com/aigamedev/resistance), as they were developed in conjunction to them.

# What does the agent need?

To be able to work the agent should needs access to the following object:

- `spy_classifier`: that's the NN model
- `vote_info_file.csv`: containing the normalized data for the decision tree
- `orig_vote_info_file.csv`: containing the non-normalized data for the decision tree
- `orig_LoggerBot`: containing the non-normalized data for the NN model

In here I assume that all of the objects will be contained in the `resources` folder, and that this object will be in the folder of `bots/agent/`. If that's not the case changing the variable of `path_to_folder` should do the trick
