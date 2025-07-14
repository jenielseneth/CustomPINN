import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.layers = [torch.nn.Linear(1, hidden_size)]
        self.layers.extend([torch.nn.Linear(hidden_size*2**i, hidden_size*2**(i+1)) for i in range(layers)])
        self.layers = torch.nn.ModuleList(self.layers)
        self.final_layer = torch.nn.Linear(hidden_size*2**layers, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return self.final_layer(x)
    
x = (torch.linspace(0.00001, 50, 1000)[...,None]).to("mps")
model = MLP(16, 8)
model.to(torch.device("mps"))
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for i in tqdm(range(1000)):
    y = model(x)
    loss = loss_func(torch.log(x), y)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = x.cpu()
plt.plot(x.detach().numpy(), np.log(x.detach().numpy()))
plt.plot(x.detach().numpy(), model(y).cpu().detach().numpy())
plt.show()