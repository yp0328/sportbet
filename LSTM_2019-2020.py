#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('filtered4_df.csv')


# In[3]:


df.columns.values


# In[4]:


pip install torch


# In[5]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[6]:


df = df[['PLAYER_NAME', 'POS', 'Team', 'GAME_DATE', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
       'TOV', 'PF', 'PTS', 'PLUS_MINUS',
       'total_fantasy_points']]

# encode player positions
le = LabelEncoder()
df["POS"] = le.fit_transform(df["POS"])
df["Team"]= le.fit_transform(df["Team"])

# scale numerical features
numerical_features = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                      'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
                      'TOV', 'PF', 'PTS', 'PLUS_MINUS']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# sequences of games for each player
sequence_length = 50  # Set the number of past games to consider for each player
input_features = len(numerical_features) + 1  # Number of input features + position encoding
X = []
y = []

for player_name in df['PLAYER_NAME'].unique():
    player_data = df[df['PLAYER_NAME'] == player_name].reset_index(drop=True)
    if len(player_data) >= sequence_length:  # Check if the player has at least 50 games
        for i in range(len(player_data) - sequence_length):
            X.append(player_data.loc[i:i + sequence_length - 1, numerical_features + ['POS']].values)
            y.append(player_data.loc[i + sequence_length, 'total_fantasy_points'])

X = np.array(X)
y = np.array(y)


# train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# LSTM model 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# parameters
hidden_size = 64
num_layers = 2
output_size = 1

# model, loss function, optimizer
model = LSTMModel(input_features, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[8]:


def regression_accuracy(y_true, y_pred, threshold=0.3):
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    within_threshold = torch.abs(y_true - y_pred) <= (threshold * y_true)
    accuracy = torch.mean(within_threshold.type(torch.float32))
    return accuracy.item()


# In[9]:


num_epochs = 100
batch_size = 8
train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

for epoch in range(num_epochs):
    epoch_loss = 0
    n_batches = 0
    for i in range(0, len(train_tensor), batch_size):
        batch_X = train_tensor[i:i + batch_size]
        batch_y = y_train_tensor[i:i + batch_size]

        # forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        n_batches += 1

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # average loss for epoch
    avg_epoch_loss = epoch_loss / n_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor)
        test_accuracy = regression_accuracy(y_test_tensor, y_pred)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
    # train
    model.train()


# In[10]:


from sklearn.metrics import mean_absolute_error, r2_score

y_pred_np = y_pred.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()

mae = mean_absolute_error(y_test_np, y_pred_np)
r2 = r2_score(y_test_np, y_pred_np)

print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')


# ## Now that we have trained the LSTM model, let us construct a function that predicts the total fantasy points that each player will generate for their next future game. 

# In[11]:


sequence_length = 50

player_names_50 = []
for player_name in df['PLAYER_NAME'].unique():
    player_data = df[df['PLAYER_NAME'] == player_name].reset_index(drop=True)
    if len(player_data) >= sequence_length:
        player_names_50.append(player_name)

player_names_50 


# In[12]:


seq_lengths = []
for player_name in df['PLAYER_NAME'].unique():
    player_data = df[df['PLAYER_NAME'] == player_name]
    seq_lengths.append({'PLAYER_NAME': player_name, 'SEQ_LENGTH': len(player_data)})
seq_lengths_df = pd.DataFrame(seq_lengths)
seq_lengths_df


# In[13]:


player_name = 'Aaron Gordon'
player_points_sequence = df.loc[df['PLAYER_NAME'] == player_name, 'total_fantasy_points'].tolist()
player_points_sequence


# ## example of one player's predicted total fantasy points for their next game

# In[14]:


# drop na values
seq_lengths_df = seq_lengths_df[(seq_lengths_df != 0).all(1)].dropna()

# most recent sequence of games
player_name = 'Aaron Gordon'
player_data = df[df['PLAYER_NAME'] == player_name].reset_index(drop=True)

# player max sequence length from seq_lengths_df
sequence_length = seq_lengths_df.loc[seq_lengths_df['PLAYER_NAME'] == player_name, 'SEQ_LENGTH'].item()
last_sequence = player_data.iloc[-sequence_length:, :]

# numerical features scaled
last_sequence[numerical_features] = scaler.transform(last_sequence[numerical_features])

# convert to tensor and predict future total fantasy points for next game
with torch.no_grad():
    input_tensor = torch.tensor(last_sequence[numerical_features + ['POS']].values.reshape(1, -1, input_features),
                                dtype=torch.float32)
    pred = model(input_tensor).item()
    print(f'Predicted total fantasy points for {player_name} in the next game: {pred:.2f}')


# ## function that predicts all players total fantasy points for their next game

# In[15]:


def predict_next_game_scores(df, seq_lengths_df, numerical_features, input_features, scaler, model, unique_players):
    player_names = []
    predicted_scores = []
    
    for player_name in unique_players:
        player_data = df[df['PLAYER_NAME'] == player_name].reset_index(drop=True)
        
        if not player_data.empty:
            sequence_length = seq_lengths_df.loc[seq_lengths_df['PLAYER_NAME'] == player_name, 'SEQ_LENGTH'].item()
            last_sequence = player_data.iloc[-sequence_length:, :]
            last_sequence[numerical_features] = scaler.transform(last_sequence[numerical_features])
            
            
            with torch.no_grad():
                input_tensor = torch.tensor(last_sequence[numerical_features + ['POS']].values.reshape(1, -1, input_features),
                                            dtype=torch.float32)
                pred = model(input_tensor).item()
                
            player_names.append(player_name)
            predicted_scores.append(pred)
        else:
            player_names.append(player_name)
            predicted_scores.append(None)

    predictions_df = pd.DataFrame({'PLAYER_NAME': player_names, 'PRED_SCORE': predicted_scores})
    return predictions_df


unique_players = df['PLAYER_NAME'].unique()


player_predictions_df = predict_next_game_scores(df, seq_lengths_df, numerical_features, input_features, scaler, model, unique_players)


print(player_predictions_df)


# In[16]:


filtered4_prediction = player_predictions_df
print(filtered4_prediction)
filtered4_prediction.to_csv('filtered4_prediction.csv')


# In[ ]:




