from torch.autograd import Function

#This got repetitive because PyTorch requires the forward/backward to be static methods

class AlertGradUpdate(Function):
    # def __init__(self, message):
    #     super().__init__()
    #     self.message = message
    @staticmethod
    def forward(ctx, tensor):
        # ctx is a context object that can be used to stash information
        # for backward computation
        return tensor
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Alert_View(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        print(">>>View grads updated<<<")
        return grad_output, None

class Alert_Light(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        print(">>>Light grads updated<<<")
        return grad_output, None

class Alert_Depth(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        print(">>>Depth grads updated<<<")
        return grad_output, None

class Alert_Albedo(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        print(">>>Albedo grads updated<<<")
        return grad_output, None
