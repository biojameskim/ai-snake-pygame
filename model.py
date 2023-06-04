import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    # In the constructor, declare all the layers you want to use.
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # super() allows you to use methods from nn.Module.
        # The methods from nn.Module that we used are the __init__() and forward() methods.

        # Two linear layers in the Neural Network
        # input layer
        self.linear1 = nn.Linear(input_size, hidden_size) # (input size, output size)
        # hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size) # (input size, output size)
    
    # In the forward function, define how your model is going to be run, from input to output
    # It's the forward function that defines the network structure
    def forward(self, input): # x is a tensor
        x = F.relu(self.linear1(input)) # relu is an activation function
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Loss function

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float) # convert state to tensor
        next_state = torch.tensor(next_state, dtype=torch.float) # convert next_state to tensor
        action = torch.tensor(action, dtype=torch.float) # convert action to tensor
        reward = torch.tensor(reward, dtype=torch.float) # convert reward to tensor

        if len(state.shape) == 1: # if state is a 1D tensor, we need to add a dimension to it
            state = torch.unsqueeze(state, 0) # unsqueeze adds a dimension to the tensor
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, ) # tuple with only one element

        # 1: Get predicted Q values with current state
        pred = self.model(state)

        target = pred.clone() # clone the predicted Q values
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not game_over
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # loss = MSE(Q, Q_new)
        self.optimizer.zero_grad() # set gradients to zero
        loss = self.criterion(target, pred) # calculate loss
        loss.backward() # calculate gradients

        self.optimizer.step() # update weights