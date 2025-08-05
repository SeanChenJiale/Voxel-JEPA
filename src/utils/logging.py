import logging                     # Python built-in logging library for logs
import sys                         # Used to direct logs to standard output
import torch                       # PyTorch library for GPU timing and tensor operations


def gpu_timer(closure, log_timings=True):  # Times GPU execution of a function `closure`
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)  # CUDA start event
        end = torch.cuda.Event(enable_timing=True)    # CUDA end event
        start.record()                                # Start recording

    result = closure()                                # Execute target function

    if log_timings:
        end.record()                                  # Stop recording
        torch.cuda.synchronize()                      # Sync for accurate timing
        elapsed_time = start.elapsed_time(end)        # Compute elapsed time

    return result, elapsed_time                       # Return result and timing

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"  # Log format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"                                           # Date format for logs

def get_logger(name=None, force=False):            # Sets up and returns a logger
    logging.basicConfig(stream=sys.stdout,         # Logs to stdout
                        level=logging.INFO,        # Default level: INFO
                        format=LOG_FORMAT,         # Use defined format
                        datefmt=DATE_FORMAT,       # Use defined date format
                        force=force)               # Override existing config
    return logging.getLogger(name=name)            # Return logger instance

class CSVLogger(object):                           # Custom class to log values into CSV files
    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        with open(self.fname, '+a') as f:          # Open file in append mode
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])            # Store format string
                if i < len(argv):
                    print(v[1], end=',', file=f)    # Write header
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):                          # Logs one row of values to CSV
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

class AverageMeter(object):                        # Tracks running average, min, max
    def __init__(self):
        self.reset()

    def reset(self):                               # Reset tracking
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):                    # Update with new value
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def grad_logger(named_params):                    # Logs gradient stats of model
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:                         # Track qkv attention grads
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

def adamw_logger(optimizer):                      # Logs optimizer's moment stats
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}

# Extension of CSVLogger to integrate wandb logging

class WandBCSVLogger(CSVLogger):                  # Logs training stats to CSV and wandb
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.fields = [
            ('%d', 'epoch'),
            ('%d', 'itr'),
            ('%.5f', 'loss'),
            ('%.5f', 'loss-jepa'),
            ('%.5f', 'reg-loss'),
            ('%.5f', 'enc-grad-norm'),
            ('%.5f', 'pred-grad-norm'),
            ('%d', 'gpu-time(ms)'),
            ('%d', 'wall-time(ms)')
        ]
        super().__init__(csv_path, *self.fields)
        import wandb
        self.wandb = wandb

    def log(self, *args):                         # Log both to wandb and CSV
        keys = [k for (_, k) in self.fields]
        self.wandb.log(dict(zip(keys, args)))
        super().log(*args)

class WandBCSVLoggerEval(CSVLogger):              # Eval logger variant for wandb
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.fields = [
            ('%d', 'epoch'),
            ('%.5f', 'loss'),
            ('%.5f', 'acc')
        ]
        super().__init__(csv_path, *self.fields)
        import wandb
        self.wandb = wandb

    def log(self, *args):                         # Log to wandb and CSV
        keys = [k for (_, k) in self.fields]
        self.wandb.log(dict(zip(keys, args)))
        super().log(*args)

# class WandBCSVLoggerEval(CSVLogger):
#     def __init__(self, csv_path):
#         self.csv_path = csv_path
#         self.fields = [
#             ('%d', 'epoch'),
#             ('%.5f', 'loss'),
#             ('%.5f', 'acc')
#         ]
#         super().__init__(csv_path, *self.fields)
#         import wandb
#         self.wandb = wandb

#     def log(self, epoch, loss, acc):
#         metrics_dict = {
#             'epoch': epoch,
#             'loss': loss,
#             'acc': acc
#         }
#         self.wandb.log(metrics_dict)
#         super().log(epoch, loss, acc)


class WandBCSVLogger2(CSVLogger):                 # Generic wandb logger with dynamic fields
    def __init__(self, csv_path, fields):
        self.csv_path = csv_path
        self.fields = fields
        super().__init__(csv_path, *fields)
        import wandb
        self.wandb = wandb

    def log(self, *args):                         # Log to wandb and CSV
        keys = [k for (_, k) in self.fields]
        self.wandb.log(dict(zip(keys, args)))
        super().log(*args)

def init_csv_writer(file_path):
    import csv
    import os # for init_csv_writer
    """
    Initialize a CSV writer and return the writer object.
    
    Args:
        file_path (str): Path to the CSV file to write to.
    
    Returns:
        csv.writer: A CSV writer object.
        file: The opened file object (to ensure it stays open).
    """
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path),exists_ok=True)
    
    file = open(file_path, mode='w', newline='')
    writer = csv.writer(file)
    return writer, file