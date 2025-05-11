from torch.utils.data import Dataset
import torch

class Dataset:
    class CollocationTrainingDataset(Dataset):
        def __init__(self, file_path: str):
            self.data = torch.load(file_path)
            self.coordinates = self.data["collocation_data"]["coordinates"]
            self.values = self.data["collocation_data"]["values"]
            self.length = len(self.coordinates)
            # load the images from file

        def __len__(self):
            # return total dataset size
            return self.length

        def __getitem__(self, index):
            # write your code to return each batch element
            return self.coordinates[index], self.values[index]
    
    class BoundaryTrainingDataset(Dataset):
        def __init__(self, file_path: str):
            self.data = torch.load(file_path)
            self.coordinates = self.data["boundary_data"]["coordinates"]
            self.values = self.data["boundary_data"]["values"]
            self.length = len(self.coordinates)
            # load the images from file

        def __len__(self):
            # return total dataset size
            return self.length

        def __getitem__(self, index):
            # write your code to return each batch element
            return self.coordinates[index], self.values[index]
        
    def __init__(self, file_path:str):
        self.file_path= file_path
        self.collocation_dataset = self.CollocationTrainingDataset(file_path=file_path)
        self.boundary_dataset = self.BoundaryTrainingDataset(file_path=file_path)
    
class TestDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)
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
        print(f"\rAvg Train Loss per sample: {loss:>7f}  [{current_num:>5d}/{size:>5d}] \n", end="")

def test(dataloader, model, loss_fn, f_source_term, domain):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    # with torch.no_grad():
    for coordinate, value in dataloader:
        test_loss +=  loss_fn(greens_function_approx=model, f_source_term=f_source_term, coordinates=coordinate, domain=domain,u=value).item() * len(coordinate)

    print(f"Avg Test Loss per sample: {test_loss / size :>8f} \n", end="")
    return test_loss


