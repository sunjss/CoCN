import torch


class Hsigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=10):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        y = torch.where((x > 0), torch.ones_like(x), torch.zeros_like(x))
        return y

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        y = 1 / (1 + torch.exp(- epsilon * x))
        dx = dy * epsilon * y * (1 - y)
        return dx, None


def diag_unfold(adj, filter_size, stride):
    b_s, h, d, n = adj.shape[:-1]
    idx = torch.arange(0, filter_size, device=adj.device)
    x = idx.view(1, -1, 1).expand(1, filter_size, filter_size)
    remove_idx = (x != x.permute(0, 2, 1)).nonzero()

    idx_adder = torch.arange(0, n - filter_size + 1, stride, device=adj.device)
    idx_adder = idx_adder.view(-1, 1, 1)
    x = x + idx_adder
    y = x[:, remove_idx[:, 2], remove_idx[:, 1]]
    x = x[:, remove_idx[:, 1], remove_idx[:, 2]]
    
    out = adj[:, :, :, x, y]
    out = out.permute(0, 1, 3, 2, 4).flatten(-2, -1)
    return out