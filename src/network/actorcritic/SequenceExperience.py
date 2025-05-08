import numpy as np

class SequenceExperience:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, state, action, reward, next_state, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get_sequence(self):
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs)
        }