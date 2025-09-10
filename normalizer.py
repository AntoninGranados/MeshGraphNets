import torch


class Normalizer(torch.nn.Module):
    def __init__(self, device: torch.device, size: int, std_epsilon: float = 1e-6, max_accumulation: int = int(1e5)):
        super().__init__()

        self.std_epsilon = std_epsilon
        self.max_accumulation = max_accumulation

        # Values to save with the rest of the parameters
        self.register_buffer("acc_count", torch.Tensor([0]))
        self.register_buffer("acc_sum", torch.zeros(size))
        self.register_buffer("acc_sum_sqr", torch.zeros(size))
        
    def __accumulate(self, data: torch.Tensor) -> None:
        flat_data = data.reshape([-1, data.shape[-1]])
        batch_count = flat_data.shape[0]
        sum = torch.sum(flat_data, dim=0)
        sum_sqr = torch.sum(torch.square(flat_data), dim=0)

        self.acc_count = torch.add(self.acc_count, batch_count)
        self.acc_sum = torch.add(self.acc_sum, sum)
        self.acc_sum_sqr = torch.add(self.acc_sum_sqr, sum_sqr)

    def __mean(self):
        safe_count = torch.clamp(self.acc_count, min=1)
        return self.acc_sum / safe_count
    
    def __std(self):
        safe_count = torch.clamp(self.acc_count, min=1)
        var = self.acc_sum_sqr / safe_count - self.__mean()**2
        # var = torch.clamp(var, min=0.0)
        return torch.clamp(torch.sqrt(var), min=self.std_epsilon)


    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.__std() + self.__mean()

    def __call__(self, data: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        if accumulate and self.acc_count.item() < self.max_accumulation:
            self.__accumulate(data)
        return (data - self.__mean()) / self.__std()
