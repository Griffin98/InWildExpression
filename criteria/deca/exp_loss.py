
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpLoss(nn.Module):
    def __init__(self, model, type="l1"):
        super(ExpLoss, self).__init__()
        self.model = model
        self.model.eval()

        self.type = type

        if self.type == "cosine":
            self.loss = nn.CosineSimilarity()
        else:
            self.loss = nn.SmoothL1Loss()

    def get_expression(self, images):
        with torch.no_grad():
            id_codedict = self.model.encode(images)

        expression = id_codedict['exp']

        return expression

    def forward(self, images, expressions):

        predicted_expression = self.get_expression(images)

        if self.type == "cosine":
            loss = self.loss(predicted_expression, expressions)
            loss = 1 - loss
            loss = torch.mean(loss)
        else:
            loss = self.loss(predicted_expression, expressions)

        return loss

