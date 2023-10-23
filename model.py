import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Defines a neural network class named Linear_Net for Q-learning.

class Linear_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Create the first linear layer with input_size and hidden_size.
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Create the second linear layer with hidden_size and output_size.
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Defines the forward pass of the network, applying ReLU activation to the first layer.
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Defines a method to save the model's state dictionary to a file.
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Defines a QTrainer class for training the Q-learning model.

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        # Initialize an Adam optimizer with the model's parameters and the given learning rate.
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Define the mean squared error loss function.
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        # Convert input data (state, action, reward, next_state) to PyTorch tensors.
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # Ensure that the input tensors have a batch dimension.
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over,)

        # Forward pass to get Q-values for the given state.
        pred = self.model(state)

        # Compute the target Q-values for the training update.
        target = pred.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                # If the game is not over, calculate the Q-value using the Bellman equation.
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            # Update the target Q-value for the selected action.
            target[index][torch.argmax(action).item()] = Q_new

        # Zero the gradients, calculate the loss, and perform backpropagation.
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Update the model's weights using the optimizer.
        self.optimizer.step()
