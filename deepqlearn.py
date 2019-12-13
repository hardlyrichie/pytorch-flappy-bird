import os
import sys
import random
sys.path.append("game/")
import wrapped_flappy_bird as game
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import namedtuple

class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        
        # Discount factor
        self.gamma = 0.99
        # Epsilon values for ϵ greedy exploration
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.0001
        self.replay_memory_size = 10000
        self.num_iterations = 2000000
        self.minibatch_size = 32
        
        # Use gpu if it is availiable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use same network architecture as DeepMind
        # Input is 4 frames stacked to infer velocity
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1 = nn.Linear(3136, 512)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Transition that maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory:
    """A cyclic buffer of bounded size that holds the transitions observed recently"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Selects a random batch of transitions for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def image_to_tensor(image):
    """Converts image to a PyTorch tensor"""
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor

def resize_and_bgr2gray(image):
    # Crop out the floor
    image = image[0:288, 0:404]
    # Convert to grayscale and resize image
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def train(net, start):
    """ 
    Trains the Deep Q-Network
    
    Args:
        net: torch.nn model
        start: time start training

    """
    # Initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    # Initialize loss function
    loss_func = nn.MSELoss()

    # Initialize game
    game_state = game.GameState()

    # Initialize replay memory
    memory = ReplayMemory(net.replay_memory_size)

    # Initial action is do nothing
    action = torch.zeros(2, dtype=torch.float32)
    action[0] = 1

    # [1, 0] is do nothing, [0, 1] is fly up
    image_data, reward, terminal = game_state.frame_step(action)

    # Image Preprocessing
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # Initialize epsilon value
    epsilon = net.initial_epsilon

    # Epsilon annealing
    epsilon_decrements = np.linspace(net.initial_epsilon, net.final_epsilon, net.num_iterations)
    
    # Train Loop
    for iteration in range(net.num_iterations):
        # Get output from the neural network
        output = net(state)[0]

        # Initialize action
        action = torch.zeros(2, dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # Epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(2, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        # Get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # Save transition to replay memory
        memory.push(state, action, reward, state_1, terminal)

        # Epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # Sample random minibatch
        minibatch = memory.sample(min(len(memory), net.minibatch_size))

        # Unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # Get output for the next state
        output_1_batch = net(state_1_batch)

        # Set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + net.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # Extract Q-value (this part i don't understand)
        q_value = torch.sum(net(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()

        # Returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # Calculate loss
        loss = loss_func(q_value, y_batch)

        # Do backward pass
        loss.backward()
        optimizer.step()

        # Set state to be state_1
        state = state_1

        if iteration % 25000 == 0:
            torch.save(net, "model_weights/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == 'test':
        pass

    elif mode == "train":
        if not os.path.exists('model_weights/'):
            os.mkdir('model_weights/')

        Q = QNetwork()
        Q.to(Q.device)

        start = time.time()
        train(Q, start)