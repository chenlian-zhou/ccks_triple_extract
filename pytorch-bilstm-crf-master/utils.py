import torch

CUDA = torch.cuda.is_available()


def load_checkpoint(filename, model=None):
    print("loading model...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = {}, loss = {}".format(epoch, loss))
    return epoch


def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = {}, loss = {}, time = {}".format(epoch, loss, time))
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch{}-{}".format(epoch, loss))


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x


def scalar(x):
    return x.view(-1).data.tolist()[0]
