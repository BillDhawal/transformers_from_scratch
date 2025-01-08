import torch
import torch.nn as nn
import torch.optim as optim

# Example code to understand pytorch APIs, The task here is Linear Regression.

# Sample data: [Size (sq. ft)], [Price (in $1000s)]
X = torch.tensor([[500.0], [750.0], [1000.0], [1250.0], [1500.0]])
y = torch.tensor([[100.0], [150.0], [200.0], [250.0], [300.0]])

# Normalize data
X = X / 1000
y = y / 1000

model = nn.Linear(in_features=1,out_features=1)
loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr= 0.1)

epochs = 1000
for epoch in range(epochs):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Testing the model with new data
new_X = torch.tensor([[1800.0]]) / 1000  # Normalize
predicted_price = model(new_X)
print(f"Predicted Price: ${predicted_price.item() * 1000:.2f}")


import matplotlib.pyplot as plt

# Plot predictions vs actual data
predicted = model(X).detach()
plt.scatter(X, y, label='Actual Data')
plt.plot(X, predicted, label='Fitted Line', color='red')
plt.legend()
plt.xlabel('Size (normalized)')
plt.ylabel('Price (normalized)')
plt.title('Linear Regression with PyTorch')
plt.show()
