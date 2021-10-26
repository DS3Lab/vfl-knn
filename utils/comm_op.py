import numpy as np

import torch
import torch.distributed as dist


def sum_sqrt_all_reduce(dist_arr):
    """ sum square distance and calculate sqrt """
    dist_tensor = torch.from_numpy(dist_arr)
    dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
    return np.sqrt(dist_tensor.numpy())


def sum_all_reduce(np_arr):
    """ sum square distance and calculate sqrt """
    np_tensor = torch.from_numpy(np_arr)
    dist.all_reduce(np_tensor, op=dist.ReduceOp.SUM)
    return np_tensor.numpy()


def sum_all_gather(dist_arr):
    dist_tensor = torch.from_numpy(dist_arr)
    tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, dist_tensor)
    sum_tensor = torch.stack(tensor_list, dim=0).sum(dim=0)
    return np.sqrt(sum_tensor.numpy())


def sum_gather(dist_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(dist_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        sum_tensor = torch.stack(tensor_list, dim=0).sum(dim=0)
        dist.barrier()
        return np.sqrt(sum_tensor.numpy())
    else:
        dist.gather(dist_tensor, gather_list=None)
        dist.barrier()
        return None


def gather(np_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(np_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        return [t.numpy() for t in tensor_list]
    else:
        dist.gather(dist_tensor, gather_list=None)
        return None


def gather_np(np_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(np_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        return np.array([t.numpy() for t in tensor_list])
    else:
        dist.gather(dist_tensor, gather_list=None)
        return None


if __name__ == "__main__":
    a = np.asarray([1,2,3])
    b = np.asarray([4,5,6])
    t_list = [torch.from_numpy(a), torch.from_numpy(b)]
    t_sum = torch.stack(t_list, dim=0).sum(dim=0)
    print(t_sum)
