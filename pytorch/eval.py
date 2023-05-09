# coding: utf-8
import argparse
import time
import math
import os
import sys

import torch

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--eval_warmup', type=int, default=5)
parser.add_argument('--eval_iters', type=int, default=0)
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, bfloat16')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--arch', type=str, default=None, help='model name')
parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')

args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

if args.cuda:
    device = torch.device("cuda")
elif args.ipex:
    import intel_extension_for_pytorch as ipex
    device = None
else:
    device = torch.device("cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                              device=device, ext_len=args.ext_len)

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
if args.channels_last:
    model_oob = model
    try:
        model_oob = model_oob.to(memory_format=torch.channels_last)
        print("---- Use channels last format.")
    except:
        print("---- Use normal model.")
    model = model_oob
else:
    model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
    args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

model.eval()
if args.ipex:
    if args.precision == "bfloat16":
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        print('Running with bfloat16...')
    else:
        model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        print("running fp32 evalation step\n")

if args.jit:
    jit_data = None
    jit_target = None
    for idx, (data, target, seq_len) in enumerate(te_iter):
        jit_data = data
        jit_target = target
        #ret = model(jit_data, jit_target, *jit_mems)
        model = torch.jit.trace(model, (jit_data, jit_target))
        print("---- With JIT enabled.")
        break
    if args.ipex:
        model = torch.jit.freeze(model)
    for i in range(10):
        model(jit_data, jit_target)

###############################################################################
# Evaluation code
###############################################################################


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


def evaluate(eval_iter, model):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    total_time, total_sample = 0., 0
    batch_time_list = []
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            if args.eval_iters != 0 and idx >= args.eval_iters:
                break
            if args.ipex:
                start_time = time.time()
                ret = model(data, target, *mems)
                end_time = time.time()
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
            else:
                if args.channels_last:
                    data_oob, target_oob = data, target
                    try:
                        data_oob = data_oob.to(memory_format=torch.channels_last)
                        target_oob = target_oob.to(memory_format=torch.channels_last)
                    except:
                        pass
                    data, target = data_oob, target_oob
                start_time = time.time()
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                        ret = model(data, target, *mems)
                    #
                    if idx == int(args.eval_iters / 2):
                        import pathlib
                        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                        if not os.path.exists(timeline_dir):
                            os.makedirs(timeline_dir)
                        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                    "transformerxl" + str(idx) + '-' + str(os.getpid()) + '.json'
                        print(timeline_file)
                        prof.export_chrome_trace(timeline_file)
                        table_res = prof.key_averages().table(sort_by="cpu_time_total")
                        print(table_res)
                        # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                else:
                    ret = model(data, target, *mems)
                end_time = time.time()
                print("Iteration: {}, inference time: {} sec.".format(idx, end_time - start_time), flush=True)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len

            if idx >= args.eval_warmup:
                total_time += end_time - start_time
                total_sample += args.batch_size
                batch_time_list.append((end_time - start_time) * 1000)

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("Latency:\t {:.3f} ms".format(latency))
    print("Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    return total_loss / total_len


# Run on test data.
if args.precision == "bfloat16":
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        if args.split == 'all':
            test_loss = evaluate(te_iter)
            valid_loss = evaluate(va_iter)
        elif args.split == 'valid':
            valid_loss = evaluate(va_iter)
            test_loss = None
        elif args.split == 'test':
            test_loss = evaluate(te_iter, model)
            valid_loss = None
else:
    if args.split == 'all':
        test_loss = evaluate(te_iter)
        valid_loss = evaluate(va_iter)
    elif args.split == 'valid':
        valid_loss = evaluate(va_iter)
        test_loss = None
    elif args.split == 'test':
        test_loss = evaluate(te_iter, model)
        valid_loss = None

def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
