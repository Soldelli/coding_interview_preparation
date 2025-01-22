import torch

def create_tensors():
    # manually
    x = torch.tensor([[0,1], [2, 3]], dtype = torch.float32)
    print(x.shape)

    # Zeros 
    x = torch.zeros((2, 3))
    print(x.shape)

    # Ones 
    x = torch.ones((2, 4))
    print(x.shape)

    # Random Tensors
    x = torch.randn(2, 5)
    print(x.shape)

    x = torch.randint(low=0, high=10, size=(2, 6))
    print(x.shape)
    
    # Eye (identity matrix)
    x = torch.eye(3)
    print(x.shape)
    
    # Range
    x = torch.arange(start=0, end=10, step=2)
    print(x.shape)

    # Zero/ones Like
    x = torch.zeros_like(x)
    print(x.shape)
    x = torch.ones_like(x)
    print(x.shape)

    # move to device: 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.ones((2, 2), dtype=torch.float64, device=device)
    print(x.dtype)
    print(x.device)
    print(x.shape)


def tensor_shape_operations():

    x = torch.tensor([[0,1], [2, 3]], dtype = torch.float32)

    # reshaping tensors
    xx = x.view(4, 1)
    print(xx.shape)
    xx = x.view(1, 4)
    print(xx.shape)

    # transpose dimensions
    xx = xx.transpose(1,0)
    print(xx.shape)


if __name__ == "__main__":
    create_tensors()
    # tensor_shape_operations()

