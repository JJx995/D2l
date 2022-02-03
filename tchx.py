import os
import re
import sys
import time
import math
import shutil
import random
import zipfile, tarfile
import inspect
import hashlib
import collections
import seaborn as sns

import matplotlib_inline.backend_inline
import requests

from collections import OrderedDict as ordict

from IPython import display
from ipywidgets import Output
from IPython.display import clear_output as cloutput
from matplotlib import pyplot as plts
import matplotlib as mtplib
from matplotlib import image as Imgmpl
from PIL import Image

import pandas as pd
import numpy as np

import torch as tch
import torchvision as tchvision
from torch import nn as tchnet
from torch.utils import data as Data  # noqa
from torch.nn import functional
from torchvision import transforms

Xtch = sys.modules[__name__]

# change this for running in jupyter.
Jupyternb = False
# you may have to change it to True.

# sns.set()
# sns.set_theme()
# plts.style.use('dark_background')
# plts.rcParams['axes.facecolor'] = 'none'
DATA_HUB = dict()
DATA_URL = 'https://d2l-data.s3-accelerate.amazonaws.com/'


def set_notebook_plot():
    sns.set()
    sns.set_theme()
    # plts.style.use('dark_background')  # comment this line when not using PyCharm
    plts.rcParams['axes.facecolor'] = 'none'


# caution??
def restore_rcParams():
    mtplib.rcParams.update(mtplib.rcParamsDefault)
    plts.style.use('dark_background')
    plts.rcParams['axes.facecolor'] = 'none'


def use_svg_display():
    if Jupyternb:
        matplotlib_inline.backend_inline.set_matplotlib_formats('retina')


def set_figsize(figsize=(16, 16)):
    use_svg_display()
    plts.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None,
         xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(16, 16), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)

    original = False
    if not axes:
        original = True
        fig, axes = plts.subplots()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel,
             xlim, ylim,
             xscale, yscale,
             legend)
    if original:
        return fig, axes  # noqa


class Timer:
    def __init__(self):
        self.times = []
        self.tik = None
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def load_array(data_arrays, batch_size, is_train=True):
    dataset = Data.TensorDataset(*data_arrays)
    return Data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plts.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if tch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():
    return 1


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = tchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = tchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            Data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


def accuracy(y_hat, y):
    if not tch.is_tensor(y_hat):
        y_hat = tch.tensor(y_hat)
    if not tch.is_tensor(y):
        y = tch.tensor(y)
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tch.argmax(y_hat, dim=1)
    cmp = y_hat.to(dtype=y.dtype) == y
    return float(cmp.to(dtype=y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, tchnet.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with tch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    # Set the model to training mode
    if isinstance(net, tchnet.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        ls = loss(y_hat, y)
        if isinstance(updater, tch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            ls.mean().backward()
            updater.step()
        else:
            ls.sum().backward()
            updater(X.shape[0])
        metric.add(float(ls.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(16, 16)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plts.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel,
                                            xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not isinstance(x, tch.Tensor):
            x = tch.tensor(x)
        if not isinstance(y, tch.Tensor):
            y = tch.tensor(y)

        if not x.dim():
            x = tch.tensor([x])
        n = len(x)
        if not y.dim():
            y = [y] * n
            y = tch.tensor(y)
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))  # noqa
    train_loss, train_acc = train_metrics  # noqa
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc  # noqa


def predict_ch3(net, test_iter, n=6):
    # this one is for chapter 3 in which we classify a bunch of pictures from fashion_mnist.
    # why use for? (in book it has been written with for.)
    X, y = next(iter(test_iter))
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(tch.argmax(net(X), dim=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        ls = loss(out, y)
        metric.add(ls.sum(), ls.numel())
    return metric[0] / metric[1]


def download(name, cache_dir=os.path.join('..', 'data')):
    # exactly the same as its D2l package function.
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


def try_gpu(indx=0):
    if tch.cuda.device_count() >= indx + 1:
        return tch.device(f'cuda:{indx}')
    return tch.device('cpu')


def try_all_gpus():
    devices = [tch.device(f'cuda:{i}') for i in range(tch.cuda.device_count())]
    return devices if devices else [tch.device('cpu')]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, tchnet.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(net.parameters()).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with tch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if isinstance(m, tchnet.Linear) or isinstance(m, tchnet.Conv2d):
            tchnet.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print('training on: ', device)
    net.to(device)

    optimizer = tch.optim.SGD(net.parameters(), lr=lr)
    loss = tchnet.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            ls = loss(y_hat, y)
            ls.backward()
            optimizer.step()
            with tch.no_grad():
                metric.add(ls * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_ls = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # changed five to Ten in the line below.
            if (i + 1) % (num_batches // 10) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_ls, train_acc, None))  # noqa
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))  # noqa
    print(f'loss {train_ls:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')  # noqa
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')  # noqa


class Residual(tchnet.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tchnet.Conv2d(input_channels, num_channels,
                                   kernel_size=(3, 3), padding=1, stride=(strides, strides))
        self.conv2 = tchnet.Conv2d(num_channels, num_channels,
                                   kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = tchnet.Conv2d(input_channels, num_channels, kernel_size=(1, 1), stride=(strides, strides))
        else:
            self.conv3 = None
        self.bn1 = tchnet.BatchNorm2d(num_channels)
        self.bn2 = tchnet.BatchNorm2d(num_channels)

    def forward(self, X):
        # uniting all of input_outputs based on previous conventions of the book. X, y
        y = functional.relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            X = self.conv3(X)
        y += X
        return functional.relu(y)


DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def count_corpus(tokens):
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        y = [data(j + 1) for j in initial_indices_per_batch]
        yield tch.tensor(X), tch.tensor(y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = tch.tensor(corpus[offset: offset + num_tokens])
    ys = tch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, ys = Xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        y = ys[:, i: i + num_steps]
        yield X, y


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = tch.seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: tch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, tchnet.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # D2l's custom scratch implementation.
        params = net.params
    norm = tch.sqrt(sum(tch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, Timer()
    metric = Accumulator(2)  # Sum of training loss, no. of tokens
    for X, y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, tchnet.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()  # noqa
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        ls = loss(y_hat, y.long()).mean()
        if isinstance(updater, tch.optim.Optimizer):
            updater.zero_grad()
            ls.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            ls.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(ls * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = tchnet.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    updater = tch.optim.SGD(net.parameters(), lr)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])  # noqa
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')  # noqa
    print(predict('time traveller'))
    print(predict('traveller'))


class RNNModule(tchnet.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = tchnet.Linear(self.hidden_size, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = tchnet.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = functional.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(dtype=tch.float32)
        y, state = self.rnn(X, state)
        return self.linear(y.reshape((-1, y.shape[-1]))), state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, tchnet.LSTM):
            return tch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device)
        else:
            return (tch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device),
                    tch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.hidden_size), device=device))


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(16, 16), cmap='Reds'):
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plts.subplots(num_rows, num_cols, figsize=figsize,
                              sharex='all', sharey='all', squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)

            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
            ax.grid(visible=None)
    fig.colorbar(pcm, ax=axes, shrink=0.6)  # noqa


def sequence_mask(X, valid_len, value=0.0):
    maxlen = X.size(1)
    mask = tch.arange(maxlen, dtype=tch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = tch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return functional.softmax(X.reshape(shape), dim=-1)


# region Machine_Translation
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()


def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# Missing show_list_len_pair_hist
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[ln] for ln in lines]
    lines = [ln + [vocab['<eos>']] for ln in lines]
    array = tch.tensor([truncate_pad(ln, num_steps, vocab['<pad>']) for ln in lines])

    valid_len = (array != vocab['<pad>']).to(dtype=tch.int32).sum(dim=1)

    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)

    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    data_iter = load_array(data_arrays, batch_size)

    return data_iter, src_vocab, tgt_vocab


class Encoder(tchnet.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(tchnet.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(tchnet.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tchnet.Embedding(vocab_size, embed_size)
        self.rnn = tchnet.GRU(embed_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X).permute(1, 0, 2)
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)

        return output, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tchnet.Embedding(vocab_size, embed_size)
        self.rnn = tchnet.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        # As the book states, "to further incorporate the encoded input sequence information"
        # the CONTEXT VARIABLE is concatenated with the decoder input at every(all) time steps.

        self.dense = tchnet.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # shape is (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # broadcast context, so it has the same num_steps as X
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = tch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)

        # output.shape, state.shape (batch_size, num_steps, vocab_size), (num_layers, batch_size, hidden_size)
        return output, state


class MaskedSoftmaxCELoss(tchnet.CrossEntropyLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, *args, **kwargs):
        # pred.shape (batch_size, num_steps, vocab_size)
        # label.shape .shape (batch_size, num_steps)
        # valid_len (batch_size,)
        valid_len = None
        if 'valid_len' in kwargs.keys():
            valid_len = kwargs['valid_len']
        elif len(args) == 1:
            valid_len = args[0]
        else:
            return super().forward(pred, label)

        weights = tch.ones_like(label)
        weights = sequence_mask(weights, valid_len)

        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if isinstance(m, tchnet.Linear):
            tchnet.init.xavier_uniform_(m.weight)
        if isinstance(m, tchnet.GRU):
            for param in m.flat_weights_names:
                if "weight" in param:
                    tchnet.init.xavier_uniform_(m.parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = tch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = tch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = tch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            ls = loss(Y_hat, Y, Y_valid_len)
            ls.sum().backward()  # Make the loss scalar for `backward`
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with tch.no_grad():
                metric.add(ls.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],)) # noqa
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} 'f'tokens/sec on {str(device)}') # noqa


# READ THIS ONE.
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = tch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # Add the batch axis
    enc_X = tch.unsqueeze(tch.tensor(src_tokens, dtype=tch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # Add the batch axis
    dec_X = tch.unsqueeze(tch.tensor([tgt_vocab['<bos>']], dtype=tch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).to(dtype=tch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, int(1 - len_label / len_pred)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
# endregion


class AdditiveAttention(tchnet.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout):
        super().__init__()
        self.W_k = tchnet.Linear(key_size, hidden_size, bias=False)
        self.W_q = tchnet.Linear(query_size, hidden_size, bias=False)
        self.w_v = tchnet.Linear(hidden_size, 1, bias=False)
        self.dropout = tchnet.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = tch.tanh(features)

        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores is
        # (batch_size, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(tchnet.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tchnet.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = tch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout)
        self.embedding = tchnet.Embedding(vocab_size, embed_size)
        self.rnn = tchnet.GRU(embed_size + hidden_size, hidden_size, num_layers, dropout)

        self.dense = tchnet.Linear(hidden_size, vocab_size)

        self._attention_weights = None

    def init_state(self, enc_outputs, *args, **kwargs):
        enc_valid_lens = None
        if 'enc_valid_lens' in kwargs.keys():
            enc_valid_lens = kwargs['enc_valid_lens']
        elif len(args) == 1:
            enc_valid_lens = args[0]
        else:
            print('Error')
            return -1

        outputs, hidden_state = enc_outputs
        # outputs.shape (num_steps, batch_size, hidden_size)
        # hidden_state[0].shape (num_layers, batch_size, hidden_size)

        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, X, state):
        # enc_outputs.shape (batch_size, num_steps, hidden_size)
        # hidden_state[0].shape (num_layers, batch_size, hidden_size)
        enc_outputs, hidden_state, enc_valid_lens = state
        # X.shape (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)

        outputs, self._attention_weights = [], []
        for x in X:
            # query.shape (batch_size, 1, num_hiddens)
            query = hidden_state[-1].unsqueeze(dim=1)
            # context.shape (batch_size, 1, hidden_size)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)

            x = tch.cat((context, x.unsqueeze(dim=1)), dim=1)

            # reshape x as (1, batch_size, embed_size + hidden_size)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

            outputs = self.dense(tch.cat(outputs, dim=0))
            return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights



