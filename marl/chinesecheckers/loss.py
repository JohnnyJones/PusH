import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, lambda_l2=1e-4) -> None:
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_l2 = lambda_l2
    
    def l2_loss(self, params):
        return self.lambda_l2 * sum(p.pow(2.0).sum() for p in params)
    
    def forward(self, pred_value: torch.Tensor, pred_truth, y_value, y_truth, params):      
        value_loss = self.mse_loss(pred_value, y_value)
        policy_loss = self.cross_entropy_loss(pred_truth.view(-1, 294), y_truth.view(-1, 294))
        l2 = self.l2_loss(params)
        
        loss = value_loss + policy_loss + l2
        return loss