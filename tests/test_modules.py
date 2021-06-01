import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from brc_pytorch import modules, utils

BATCH_SIZE = 16
SEQ_LEN = 32
INPUT_SIZE = 64
HIDDEN_SIZE = 128


class TestBRCCell(unittest.TestCase):
    CELL = modules.BRC

    def test_brc_cell_forward(self):
        cell = self.CELL(INPUT_SIZE, HIDDEN_SIZE)

        init_hidden = cell.init_hidden(BATCH_SIZE)
        input = torch.randn(BATCH_SIZE, INPUT_SIZE)
        output, hidden = cell(input, init_hidden)

        self.assertEqual(tuple(output.shape), (BATCH_SIZE, HIDDEN_SIZE))
        for h in hidden:
            self.assertEqual(tuple(h.shape), (BATCH_SIZE, HIDDEN_SIZE))


class TestNeuromodulatedBRCCell(TestBRCCell):
    CELL = modules.NBRC


class TestGRUCell(TestBRCCell):
    CELL = modules.GRUCell


class TestLSTMCell(TestBRCCell):
    CELL = modules.LSTMCell


class TestRNNLayer(unittest.TestCase):
    def test_brc_layer(self):
        rnn = modules.RNNLayer(modules.BRC, INPUT_SIZE, HIDDEN_SIZE)
        init_hidden = rnn.init_hidden(BATCH_SIZE)
        input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        out, hidden = rnn(input, init_hidden)

        self.assertEqual(tuple(out.shape), (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE))
        for h in hidden:
            self.assertEqual(tuple(h.shape), (BATCH_SIZE, HIDDEN_SIZE))

    def test_lstm_layer(self):
        rnn = modules.RNNLayer(modules.LSTMCell, INPUT_SIZE, HIDDEN_SIZE)
        init_hidden = rnn.init_hidden(BATCH_SIZE)
        input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
        out, hidden = rnn(input, init_hidden)

        self.assertEqual(tuple(out.shape), (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE))
        for h in hidden:
            self.assertEqual(tuple(h.shape), (BATCH_SIZE, HIDDEN_SIZE))


class TestStackedRNNLayer(unittest.TestCase):
    def test_stacked_brc(self):
        for num_layers in range(1, 4):
            rnn = modules.StackedRNN(modules.BRC, INPUT_SIZE, HIDDEN_SIZE, num_layers)
            init_hidden = rnn.init_hidden(BATCH_SIZE)
            input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
            out, hidden = rnn(input, init_hidden)

            self.assertEqual(tuple(out.shape), (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE))
            for layer in hidden:
                for h in layer:
                    self.assertEqual(tuple(h.shape), (BATCH_SIZE, HIDDEN_SIZE))

    def test_stacked_lstm(self):
        for num_layers in range(1, 4):
            rnn = modules.StackedRNN(modules.LSTMCell, INPUT_SIZE, HIDDEN_SIZE, num_layers)
            init_hidden = rnn.init_hidden(BATCH_SIZE)
            input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
            out, hidden = rnn(input, init_hidden)

            self.assertEqual(tuple(out.shape), (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE))
            for layer in hidden:
                for h in layer:
                    self.assertEqual(tuple(h.shape), (BATCH_SIZE, HIDDEN_SIZE))


class BenchmarkRNNs(unittest.TestCase):
    DEVICE = 'cuda'
    WARMUP_STEPS = 10
    BENCHMARK_STEPS = 100

    # Minimum performance ratio between our LSTM implementation and the reference one
    PERF_RATIO_THRESHOLD = 0.5

    def full_pass(self, mod, inputs):
        for p in mod.parameters():
            p.grad = None

        out, _ = mod(*inputs)
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()

    def benchmark_custom_cell(self, cell, num_layers):
        rnn = modules.StackedRNN(cell, 64, 128, num_layers).to(self.DEVICE)
        init_hidden = utils.to_device(rnn.init_hidden(32), self.DEVICE)
        input = torch.randn(128, 32, 64).to(self.DEVICE)

        for i in range(self.WARMUP_STEPS):
            self.full_pass(rnn, (input, init_hidden))

        start = time.time() 

        for i in range(self.BENCHMARK_STEPS):
            self.full_pass(rnn, (input, init_hidden))

        torch.cuda.synchronize()

        return time.time() - start

    def benchmark_ref(self, module, num_layers):
        rnn = module(512, 1024, num_layers).to(self.DEVICE)
        input = torch.randn(256, 128, 512).to(self.DEVICE)

        if module == nn.LSTM:
            init_hidden = utils.to_device([
                torch.randn(num_layers, 128, 1024),
                torch.randn(num_layers, 128, 1024)
            ], self.DEVICE)
        else:
            raise ValueError(f'Unknown module {module}')

        for i in range(self.WARMUP_STEPS):
            rnn(input, init_hidden)

        start = time.time()

        for i in range(self.BENCHMARK_STEPS):
            rnn(input, init_hidden)

        torch.cuda.synchronize()

        return time.time() - start

    def test_lstm(self):
        for num_layers in range(1, 4):
            our_time = self.benchmark_custom_cell(modules.LSTMCell, num_layers)
            ref_time = self.benchmark_ref(nn.LSTM, num_layers)

            self.assertTrue(ref_time / our_time > self.PERF_RATIO_THRESHOLD, 'Performance criteria not met')