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
        action  = self.forward(state)
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

################################# End of Part I ################################

print("============================================================================================")

################################### Training TD3 with 256 neurons###################################

####### initialize environment hyperparameters and TD3 hyperparameters ######
print("setting training environment : ")
replay_buffer_size = 1000000
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
max_training_timesteps = 100000   # break from training loop if timeteps > max_training_timesteps
max_ep_len = 100            # max timesteps in one episode, previously 225
rewards     = []
batch_size  = 128
random_seed = 0      
gamma = 0.99     
decay_rate = 0.995
epsilon = 1          

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)
capacity = 10000

env = Env()
# noise = GaussianExploration() # for continuous action spaces

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

## Note : print/save frequencies should be > than max_ep_len

###################### saving files ######################

#### saving files for multiple runs are NOT overwritten

log_dir = "TD3_files"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir_1 = log_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
if not os.path.exists(log_dir_1):
      os.makedirs(log_dir_1)
      
log_dir_2 = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
if not os.path.exists(log_dir_2):
      os.makedirs(log_dir_2)


#### get number of saving files in directory
run_num = 0
current_num_files1 = next(os.walk(log_dir_1))[2]
run_num1 = len(current_num_files1)
current_num_files2 = next(os.walk(log_dir_2))[2]
run_num2 = len(current_num_files2)


#### create new saving file for each run 
log_f_name = log_dir_1 + '/TD3_' + 'resource_allocation' + "_log_" + str(run_num1) + ".csv"
log_f_name2 = log_dir_2 + '/TD3_' + 'resource_allocation' + "_log_" + str(run_num2) + ".csv"

print("current logging run number for " + 'resource_allocation' + " : ", run_num1)
print("logging at : " + log_f_name)

#####################################################

################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "TD3_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + 'resource_allocation' + '/' 
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "TD3256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")
 
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate (actor) : ", lr_actor)
print("optimizer learning rate (critic) : ", lr_critic)

print("--------------------------------------------------------------------------------------------")
print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

#####################################################

print("============================================================================================")


################# training procedure ################
# initialize TD3 agent
value_criterion = nn.MSELoss()

value_optimizer1 = optim.Adam(value_net1.parameters(), lr=lr_critic)
value_optimizer2 = optim.Adam(value_net2.parameters(), lr=lr_critic)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_actor)

replay_buffer = ReplayBuffer(replay_buffer_size)

start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')
log_f2 = open(log_f_name2,"w+")
log_f2.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0
num_steps    = 50
log_interval = 10


while time_step <= max_training_timesteps:
    print("New training episode:")
    sleep(0.1) # we sleep to read the reward in console
    state = env.reset()
    current_ep_reward = 0
    q_values = []
    values   = []
    policies = []
    actions  = []
    rewards  = []
    masks    = []

    target_q_value, expected_q_value = 0.0, 0.0

    for step in range(max_ep_len):
        action_tens = policy_net.get_action(state)
        normalized_actions = NormalizedActions()
        action = normalized_actions._action(action_tens) + np.random.normal(0, 0.1)
        action = int(np.clip(action, 0, action_dim - 1))
        next_state, reward, done, _ = env.step(action)
        
        current_ep_reward += reward
        time_step += 1
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console
        
        reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
        mask   = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)

        replay_buffer.push(state, action, reward, next_state, done)

        if len(replay_buffer) > batch_size:
            target_q_value, expected_q_value = td3_update(step, batch_size)

        q_values.append(target_q_value)
        policies.append(target_policy_net)
        actions.append(action)
        rewards.append(reward)
        values.append(expected_q_value)
        masks.append(mask)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_f2.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f2.flush()
            print("Saving reward to csv file")
            sleep(0.1) # we sleep to read the reward in console
            log_running_reward = 0
            log_running_episodes = 0
            
        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            sleep(0.1) # we sleep to read the reward in console
            print_running_reward = 0
            print_running_episodes = 0
            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            sleep(0.1) # we sleep to read the reward in console
            torch.save(policy_net.state_dict(), checkpoint_path) 
            torch.save(value_net1.state_dict(), checkpoint_path) 
            torch.save(value_net2.state_dict(), checkpoint_path)
            print("model saved")
            print("--------------------------------------------------------------------------------------------")
        state = next_state
        
        # break; if the episode is over
        if done:
            break

    next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1
    time_step += num_steps


log_f.close()
log_f2.close()

################################ End of Part II ################################

print("============================================================================================")

