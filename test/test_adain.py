import torch

from utils.function import adaptive_instance_normalization as AdaIn

def main():
    input = torch.randn(4, 512, 4, 4)
    expression = torch.randn(4, 50)

    style = expression.unsqueeze(2).unsqueeze(3).repeat(1, 1, input.shape[2], input.shape[3])

    out = AdaIn(input, style)
    print(out.shape)



if __name__ == "__main__":
    main()