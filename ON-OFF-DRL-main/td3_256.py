import os
from datetime import datetime
from collections import deque
from collections import namedtuple
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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, error):
        # Unpack the transition
        state, action, reward, next_state, done = transition

        # Convert to NumPy arrays if needed
        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        
        if not isinstance(next_state, np.ndarray):
            next_state = next_state.cpu().numpy()

        # Recreate the transition with validated data
        transition = (state, action, reward, next_state, done)

        # Add to buffer
        priority = float((error + 1e-5) ** self.alpha)
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        # Ensure all priorities are scalar before conversion
        if not all(isinstance(p, (int, float)) for p in self.priorities):
            # print("Non-scalar values detected in priorities. Cleaning...")
            self.priorities = deque(float(p[0]) if isinstance(p, (list, np.ndarray)) else float(p) for p in self.priorities)

        # Convert to NumPy array
        priorities = np.array(self.priorities)
        if priorities.ndim != 1:
            raise ValueError(f"Invalid priorities shape: {priorities.shape}. Expected a 1D array.")

        # Calculate probabilities for sampling
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Extract components of the sampled transitions
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert states and next_states to tensors
        states = np.vstack(states)  # Stacks all states into a single NumPy array
        states = torch.FloatTensor(states).to(device)

        next_states = np.vstack(next_states)
        next_states = torch.FloatTensor(next_states).to(device)

        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        # Create a named tuple for the batch
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        batch = Transition(states, actions, rewards, next_states, dones)

        return batch, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


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


class OrnsteinUhlenbeckNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, decay_period=1000000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.decay_period = decay_period
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_action(self, action, t=0):
        sigma = self.sigma - (self.sigma) * min(1.0, t / self.decay_period)
        # Applying the Ornstein-Uhlenbeck process
        dx = self.theta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        action_with_noise = action + self.state
        return action_with_noise

def compute_td_errors(transitions):
    states = transitions.state
    actions = torch.cat(transitions.action, dim=0)
    rewards = torch.cat(transitions.reward, dim=0).squeeze(1) 
    next_states = transitions.next_state
    dones = torch.tensor(transitions.done, dtype=torch.float32) 
    with torch.no_grad():
        next_actions = target_policy_net(next_states)
        target_q_values = target_value_net1(next_states, next_actions)
        target_q_values = rewards + (1 - dones) * gamma * target_q_values
    
    current_q_values = value_net1(states, actions)
    td_errors = (target_q_values - current_q_values).abs()

    return td_errors.cpu().detach().numpy()

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
        # x = F.tanh(self.linear3(x))
        action_probs = F.softmax(self.linear3(x), dim=-1)
        return action_probs
    
    def get_action(self, state):
        action = self.forward(state)
        return action.detach()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(1)

        if action.shape[1] == 1: 
            num_classes = self.linear1.in_features - state.shape[1]
            action = F.one_hot(action.long(), num_classes=num_classes).float()
            action = action.view(state.size(0), -1)  # Reshape to [batch_size, action_dim]
        
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x.view(-1, 1)


def td3_update(step, batch_size, gamma=0.99, soft_tau=1e-2, noise_std=0.2, noise_clip=0.3, policy_update=2, target_update = 10, beta=0.4):

    batch, weights, indices = replay_buffer.sample(batch_size, beta)
    state, action, reward, next_state, done = batch

    # Validate and stack states
    if isinstance(state, list):
        state = np.stack(state)
    
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device).view(-1, 1)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    weights = weights.to(device)

    # Calculate next actions with target policy
    next_action = target_policy_net(next_state)
    noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
    noise = torch.clamp(noise, -noise_clip, noise_clip)
    next_action = normalized_actions._action(next_action + noise)

    target_q_value1 = target_value_net1(next_state, next_action).view(-1, 1)
    target_q_value2 = target_value_net2(next_state, next_action).view(-1, 1)
    target_q_value = torch.min(target_q_value1, target_q_value2)
    expected_q_value = reward + (1.0 - done) * gamma * target_q_value

    q_value1 = value_net1(state, action).view(-1, 1)
    q_value2 = value_net2(state, action).view(-1, 1)

    value_loss1 = (weights * F.mse_loss(q_value1, expected_q_value.detach(), reduction='none').squeeze()).mean()
    value_loss2 = (weights * F.mse_loss(q_value2, expected_q_value.detach(), reduction='none').squeeze()).mean()

    value_optimizer1.zero_grad()
    value_loss1.backward()
    value_optimizer1.step()

    value_optimizer2.zero_grad()
    value_loss2.backward()
    value_optimizer2.step()

    # Update priorities
    errors = torch.abs(q_value1 - expected_q_value).detach().cpu().numpy()
    replay_buffer.update_priorities(indices, errors)

    if step % policy_update == 0:
        policy_loss = -value_net1(state, action).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

    if step % target_update == 0:
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
max_ep_len = 225            # max timesteps in one episode, also try 100
rewards     = []
batch_size  = 128
random_seed = 0      
gamma = 0.99     
decay_rate = 0.995
epsilon = 1          
beta_start = 0.4  # Initial value

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)


# state space dimension
state_dim = args.n_servers * args.n_resources + args.n_resources + 1

# action space dimension
action_dim = args.n_servers

hidden_dim = 256

normalized_actions = NormalizedActions(action_dim)

value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

soft_update(value_net1, target_value_net1, soft_tau=1.0)
soft_update(value_net2, target_value_net2, soft_tau=1.0)
soft_update(policy_net, target_policy_net, soft_tau=1.0)

env = Env()
# noise = GaussianExploration()
noise = OrnsteinUhlenbeckNoise(action_dim)
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

replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)

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
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
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
        action_tens = noise.get_action(action_tens, step)
        action_tens = normalized_actions._action(action_tens).float()
        # Calculate initial TD error for priority
        with torch.no_grad():
            q_value = value_net1(state, action_tens)
            action = action_tens.multinomial(1)
            # action = torch.argmax(action_tens, dim=1)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            next_q_value = target_value_net1(next_state, target_policy_net(next_state))
            td_error = abs(reward + gamma * next_q_value - q_value).item()

        # sample for a single action
        current_ep_reward += reward
        time_step += 1
        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
        sleep(0.1) # we sleep to read the reward in console
        
        reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
        mask   = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)

        # Store transition
        replay_buffer.add((state, action, reward, next_state, done), error=td_error)

        if len(replay_buffer) > batch_size:
            transitions, weights, indices = replay_buffer.sample(batch_size, beta=beta_start)
            # After training, update priorities in replay buffer
            new_errors = compute_td_errors(transitions)
            replay_buffer.update_priorities(indices, new_errors)
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
        
        # if time_step % log_interval == 0:
        #     avg_reward = np.mean(rewards[-log_interval:])
        #     print(f"Step {time_step}, Average Reward: {avg_reward}")
            
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

    # next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
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