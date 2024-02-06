import torch


class NBLoss(torch.nn.Module):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self, mask=False, debug=False, device='cpu') -> None:
        super().__init__()

        self.eps = 1e-10
        self.debug = debug
        self.mask = mask
        self.device = device

    def forward(self, x, mean, theta, red_mean=True):
        if self.mask:
            x = torch.nan_to_num(x, posinf=float('inf'), neginf=-float('inf'))
        
        theta = torch.minimum(theta, torch.tensor([1e6]).to(self.device))

        t1 = torch.lgamma(theta+self.eps) + torch.lgamma(x+1.0) - torch.lgamma(x+theta+self.eps)
        t2 = (theta+x) * torch.log(1.0 + (mean/(theta+self.eps))) + (x * (torch.log(theta+self.eps) - torch.log(mean+self.eps)))

        if self.debug:
            print('No debugging implemented as of yet')
            final = t1 + t2
        else:
            final = t1 + t2
        
        final = torch.nan_to_num(final, nan=float('inf'), posinf=float('inf'), neginf=-float('inf'))

        if red_mean:
            if self.mask:
                final = torch.nanmean(final)
            else:
                final = torch.mean(final)
        
        return final


class ZINBLoss(NBLoss):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self, ridge_lambda=0.0, **kwargs) -> None:
        super().__init__(self, **kwargs)
        self.ridge_lambda = ridge_lambda
    
    def forward(self, x, mean, theta, pi, red_mean=True):
        nb_case = super().forward(x, mean, theta, red_mean=False) - torch.log(1.0-pi+self.eps)

        theta = torch.minimum(theta, torch.tensor([1e6]).to(self.device))

        zero_nb = torch.pow(theta/(theta+mean+self.eps), theta)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+self.eps)
        result = torch.where(torch.less(x, torch.tensor([1e-8]).to(self.device)), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        if red_mean:
            if self.mask:
                result = torch.nanmean(result)
            else:
                result = torch.mean(result)
        
        result = torch.nan_to_num(result, nan=float('inf'), posinf=float('inf'), neginf=-float('inf'))

        if self.debug:
            print('Implement debug stuff!')
        
        return result