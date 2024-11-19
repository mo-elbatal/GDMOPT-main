
class OrnsteinUhlenbeckNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.01, dt=1e-2, decay_period=1000000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.decay_rate = 1.0/decay_period
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_action(self, action, t=0):
        # sigma = self.sigma - (self.sigma) * min(1.0, t / self.decay_period)
        sigma = max(0.005, self.sigma - self.decay_rate * t)
        # Applying the Ornstein-Uhlenbeck process
        dx = self.theta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        state_tens = torch.tensor(self.state, dtype=action.dtype, device=action.device)
        action_with_noise = action + state_tens
        return action_with_noise
    
def compute_td_errors(transitions):
    states, actions, rewards, next_states, dones = transitions
    # print("at start of compute_td_errors, state", states.shape, "action", actions.shape, "reward", rewards.shape, "next_state", next_states.shape, "done", type(done))
    rewards = rewards.unsqueeze(-1) 
    dones = dones.unsqueeze(-1)

    with torch.no_grad():
        next_actions = target_policy_net(next_states)
        target_q_values1 = target_value_net1(next_states, next_actions)
        target_q_values2 = target_value_net2(next_states, next_actions)
        target_q_values = torch.min(target_q_values1, target_q_values2)
        target_q_values = rewards + (1 - dones) * gamma * target_q_values
    
    current_q_values1 = value_net1(states, actions)
    current_q_values2 = value_net1(states, actions)
    current_q_values = torch.min(current_q_values1, current_q_values2)

    td_errors = (target_q_values - current_q_values).abs()

    return td_errors.cpu().detach().numpy().flatten()

class NormalizedActions:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.low = 0
        self.high = action_dim - 1

    def _action(self, action_prob):
        action = self.low + (action_prob + 1.0) * 0.5 * (self.high - self.low)
        action = torch.clamp(action, min=self.low, max=self.high)
        return action

    def _reverse_action(self, action_tens):
        action = action_tens.argmax(dim=-1)
        action = 2 * (action - self.low) / (self.high - self.low) - 1
        action = torch.clamp(action, self.low, self.high)
        return action
    
