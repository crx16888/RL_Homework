# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, mask, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []