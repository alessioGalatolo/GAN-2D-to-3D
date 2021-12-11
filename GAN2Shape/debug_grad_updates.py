from torch.autograd import Function
import logging

# This got repetitive because PyTorch requires the forward/backward to be static methods


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


# FIXME: Is it possible to group them into one?
class AlertView(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        logging.info(">>>View grads updated<<<")
        return grad_output, None


class AlertLight(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        logging.info(">>>Light grads updated<<<")
        return grad_output, None


class AlertDepth(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        logging.info(">>>Depth grads updated<<<")
        return grad_output, None


class AlertAlbedo(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        logging.info(">>>Albedo grads updated<<<")
        return grad_output, None


class AlertOffsetEncoder(AlertGradUpdate):
    @staticmethod
    def backward(ctx, grad_output):
        logging.info(">>>OffsetEncoder grads updated<<<")
        return grad_output, None
