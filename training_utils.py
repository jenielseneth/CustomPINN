from torch.utils.data import Dataset
import torch

class TrainingDataset(Dataset):
    def __init__(self):
        self.data = torch.load('./data/chebyshev_uvalues_w_bc.pt')
        self.coordinates = self.data["coordinates"]
        self.values = self.data["values"]
        self.length = len(self.coordinates)
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.values[index]
    
class TestDataset(Dataset):
    def __init__(self):
        self.data = torch.load('./data/random_uvalues_test.pt')
        self.coordinates = self.data["coordinates"]
        self.values = self.data["values"]
        self.length = len(self.data["coordinates"])
        # load the images from file

    def __len__(self):
        # return total dataset size
        return self.length

    def __getitem__(self, index):
        # write your code to return each batch element
        return self.coordinates[index], self.values[index]

def train(model, optimizer, dataloader, loss_fn, f_source_term, domain, scheduler = None):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    current_num = 0
    for batch, (coordinate, value) in enumerate(dataloader):
        # Compute prediction and loss
        loss = loss_fn(greens_function_approx=model, f_source_term=f_source_term, coordinates=coordinate, domain=domain,u=value)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss = loss.item()
        total_loss += loss
        current_num= len(coordinate) + current_num
        print(f"\rloss: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")

def test(dataloader, model, loss_fn, f_source_term, domain):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    # with torch.no_grad():
    for coordinate, value in dataloader:
        test_loss +=  loss_fn(greens_function_approx=model, f_source_term=f_source_term, coordinates=coordinate, domain=domain,u=value).item()

    print(f"Avg Test loss per sample: {test_loss /size:>8f} , Avg Test loss per batch: {test_loss /num_batches:>8f} \n", end="")
    return test_loss


