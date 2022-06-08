import torch
from torch.optim import optimizer


class MyOpt(optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0):
        if momentum <= 0 or dampening != 0:
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if lr <= 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        super(MyOpt, self).__init__(params, lr, momentum, dampening, weight_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss
