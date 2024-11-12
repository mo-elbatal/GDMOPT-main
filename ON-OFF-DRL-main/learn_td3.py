import os
from datetime import datetime
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import Env
from argparser import args
from time import sleep


################################## set device to cpu or cuda ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action.squeeze(), reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class NormalizedActions:
    def _action(self, action_probs):
        low  = 0
        high = action_dim-1
        action_dist = torch.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action_tens):
        low  = 0
        high = action_dim-1
        action = action_tens.multinomial(1)
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

# used with continuous action space
# class GaussianExploration(object):
#     def __init__(self, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
#         self.low  = 0
#         self.high = action_dim
#         self.max_sigma = max_sigma
#         self.min_sigma = min_sigma
#         self.decay_period = decay_period
    
#     def get_action(self, action, t=0):
#         sigma  = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
#         action = action + np.random.normal(size=len(action)) * sigma
#         return np.clip(action, self.low, self.high)

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = tanh(self.linear3(x)  # tanh function used for continuous-action space values [-1, 1] - interferes with probs
        x = self.softmax(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action
        # return action.detach().cpu().numpy()[0]

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        action = F.one_hot(action, num_classes=action_dim).float()  # One-hot encode the action
        action = action.view(state.size(0), -1) # Reshape to (batch_size, action_dim)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.view(-1, 1)

def td3_update(step,
                batch_size,
                gamma=0.99,
                soft_tau=1e-2,
                noise_std=0.2,
                noise_clip=0.5,
                policy_update=2):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # Convert all to tensors and move to device
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).to(device)  # For discrete actions, use LongTensor
    reward = torch.FloatTensor(reward).squeeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # Policy network predicts probabilities for each action
    next_action_probs = target_policy_net(next_state)
    next_action_dist = torch.distributions.Categorical(probs=next_action_probs)
    next_action = next_action_dist.sample()

    # Add noise to target actions for exploration, clipped within limits
    noise = torch.normal(0, noise_std, size=next_action.shape).to(device)
    noise = torch.clamp(noise, -noise_clip, noise_clip)
    next_action = torch.clamp(next_action + noise, 0, action_dim - 1).long()

    # Get Q-value estimates from target networks, selecting based on `next_action`

    target_q_value1 = target_value_net1(next_state, next_action)
    target_q_value2 = target_value_net2(next_state, next_action)
    target_q_value = torch.min(target_q_value1, target_q_value2)
    
    # Calculate the expected Q-value
    expected_q_value = reward + (1.0 - done) * gamma * target_q_value

    # Calculate Q-values for the current state and chosen action
    q_value1 = value_net1(state, action)
    q_value2 = value_net2(state, action)

    q_value1 = q_value1.view(-1, 1)
    q_value2 = q_value2.view(-1, 1)
    expected_q_value = expected_q_value.view(-1, 1)

    # Calculate losses for the value networks
    value_loss1 = F.mse_loss(q_value1, expected_q_value.detach())
    value_loss2 = F.mse_loss(q_value2, expected_q_value.detach())

    # Optimize the value networks
    value_optimizer1.zero_grad()
    value_loss1.backward()
    value_optimizer1.step()

    value_optimizer2.zero_grad()
    value_loss2.backward()
    value_optimizer2.step()

    # Policy update at intervals
    if step % policy_update == 0:
        # Compute policy loss using action values from value_net1
        action_probs = policy_net(state)
        action_dist = torch.distributions.Categorical(probs=action_probs)
        chosen_action = action_dist.sample()
        
        policy_loss = -value_net1(state, action).mean()

        # Optimize the policy network
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Soft update for target networks
        soft_update(value_net1, target_value_net1, soft_tau=soft_tau)
        soft_update(value_net2, target_value_net2, soft_tau=soft_tau)
        soft_update(policy_net, target_policy_net, soft_tau=soft_tau)
    
    return target_q_value, expected_q_value

    
def load(policy_net, value_net1, value_net2, checkpoint_path):
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    value_net1.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    value_net2.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

################################# End of Part I ################################

print("============================================================================================")

replay_buffer_size = 1000000
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
rewards     = []

# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

hidden_dim = 256

value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

soft_update(value_net1, target_value_net1, soft_tau=1.0)
soft_update(value_net2, target_value_net2, soft_tau=1.0)
soft_update(policy_net, target_policy_net, soft_tau=1.0)

value_optimizer1 = optim.Adam(value_net1.parameters(), lr=lr_critic)
value_optimizer2 = optim.Adam(value_net2.parameters(), lr=lr_critic)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_actor)

replay_buffer = ReplayBuffer(replay_buffer_size)

torch.manual_seed(0)
# preTrained weights directory
random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num
directory = "TD3_preTrained" + '/' + 'resource_allocation' + '/' 
checkpoint_path = directory + "TD3256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)
# load(policy_net, value_net1, value_net2, checkpoint_path)

max_ep_len = 100            # max timesteps in one episode, previously 225

env = Env()
state = env.reset()

class learn_td3(object):
    
    def __init__(self):
        self.name = 'TD3'
        
    def step(self, obs):
        state = obs
        done = False
        total_reward = 0
        for t in range(1, max_ep_len+1):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            action_tens = policy_net.get_action(state)
            normalized_actions = NormalizedActions()
            action = normalized_actions._action(action_tens)
            action = int(action.item())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
    
        return action