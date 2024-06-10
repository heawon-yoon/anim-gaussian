import torch
import torch.nn as nn
import smplx

class ScaleActivation(nn.Module):
    def __init__(self):
        super(ScaleActivation, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)  # 将sigmoid的输出从(0, 1)缩放至(1, 10)

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        # 将Tanh的输出范围从[-1, 1]放大到[-3, 3]
        return 3.0 * torch.tanh(x)

class FitShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.beta = nn.Parameter(torch.zeros([1,10]).to(self.device).requires_grad_(True))
        self.activation = CustomActivation()
        self.sc_activation = ScaleActivation()
        self.model = smplx.create('human_model_files', model_type='smpl').to(self.device).eval()
        body_pose = torch.zeros(69, dtype=torch.float32, device=self.device)
        body_pose[2] = 0.05
        body_pose[5] = -0.05

        self.body_pose = body_pose
    def forward(self):
        x = self.activation(self.beta)
        output = self.model(betas=x, body_pose=self.body_pose[None])

        return output