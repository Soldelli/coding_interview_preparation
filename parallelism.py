import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = nn.Linear(10, 10).cuda(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(5):
        inputs = torch.randn(20, 10).cuda(rank)
        labels = torch.randn(20, 10).cuda(rank)

        outputs = ddp_model(inputs)
        loss = nn.MSELoss()(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
