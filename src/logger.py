import numpy as np
import os
import datetime
import warnings

from thirdparty.timer import Timer


def create_plot_window(vis, xlabel, ylabel, title, env):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), env=env,
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))


def create_plot_window_two(vis, xlabel, ylabel, title, legends, env):
    return vis.line(X=np.array([[1, 1]]), Y=np.array([[np.nan, np.nan]]), env=env,
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, markers=['o', 'o'], legend=legends))


def dict2str(d, start_n=0):
    res = ""
    prefix_val = " " * start_n
    for k in d:
        if type(d[k]) is dict:
            res += prefix_val + str(k) + ": " + "\n" + dict2str(d[k], start_n + 2)
        else:
            res += prefix_val + str(k) + ": " + str(d[k]) + "\n"
    return res


class Logger(object):
    def __init__(self, params):

        self.with_visdom = params['LOG']['visdom']
        self.with_tensorboard = params['LOG']['tensorboard']

        self.experiment_name = params['experiment_name']
        self.iter_interval = params['LOG']['iter_interval']
        self.num_epochs = params['TRAIN']['epochs']

        path_log_files = os.path.join(params['path_save'], self.experiment_name)

        # log files names
        filename_log_epoch = path_log_files + "_log_epoch.txt"
        filename_log_iter = path_log_files + "_log_iter.txt"
        filename_global = path_log_files + "_log.txt"

        # create and open log files
        self.f_log_iter = open(filename_log_iter, "w+")
        self.f_log_epoch = open(filename_log_epoch, "w+")
        self.f_log_global = open(filename_global, "w+")

        # init all files
        self.f_log_iter.write("{:>6} {:>14} {:>14}\n".format('iter', 'loss', 'elapsed_time'))
        self.f_log_iter.flush()
        self.f_log_epoch.write("{:>6} {:>14} {:>14} {:>14} {:>14} {:>14}\n".
                               format('epoch',
                                      'avg_acc_train', 'avg_loss_train', 'avg_acc_val', 'avg_loss_val',
                                      'elapsed_time'))
        self.f_log_epoch.flush()

        now = datetime.datetime.now()
        str2log = str(now) + "\n\n" + "==== PARAMETERS:\n" + dict2str(params)
        self.log_global(str2log)

        # timer
        self.timers = {'global': Timer()}
        self.timers['global'].tic()

        # make visdom logging
        if self.with_visdom:

            import visdom

            self.writer_vis = visdom.Visdom()
            if not self.writer_vis.check_connection():
                self.with_visdom = False
                warnings.warn("WARNING: Visdom server not running. Please run python -m visdom.server")
            else:
                self.writer_vis.close(win=None, env=self.experiment_name)

                self.train_loss_window = create_plot_window(self.writer_vis, '#Iterations', 'Loss',
                                                            'Training Loss', env=self.experiment_name)
                self.avg_loss_window = create_plot_window_two(self.writer_vis, '#Epochs', 'Loss',
                                                              'Average Loss', ['train', 'test'],
                                                              env=self.experiment_name)
                self.avg_acc_window = create_plot_window_two(self.writer_vis, '#Epochs', 'Accuracy',
                                                             'Average Accuracy', ['train', 'test'],
                                                             env=self.experiment_name)
        # make TensorBoard logging
        if self.with_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            # set-up writer
            self.writer_tb = SummaryWriter(params['path_save'])

    def log_iter(self, iter_current, epoch_current, num_iter, loss, time_str):

        if iter_current % self.iter_interval == 0:

            # log details
            log_string = "[{}] Epoch[{:^5}/{:^5}] Iteration[{:^5}/{:^5}] Loss: {:.4f} Time: {}" \
                         "".format(self.experiment_name, epoch_current, self.num_epochs,
                                   iter_current, num_iter, loss, time_str)
            self.log_global(log_string)

            globaliter = iter_current + epoch_current * num_iter

            self.f_log_iter.write(
                "{:6d} {:14.4f} {:>14}\n".format(globaliter, loss, time_str))
            self.f_log_iter.flush()

            # visdom log
            if self.with_visdom:
                self.writer_vis.line(X=np.array([globaliter]),
                                     Y=np.array([loss]),
                                     update='append', win=self.train_loss_window, env=self.experiment_name)

            # tb log
            if self.with_tensorboard:
                self.writer_tb.add_scalar('Train/RunningLoss', loss, globaliter)

    def log_epoch(self,
                  n_epoch,
                  acc_train, loss_train, acc_val, loss_val, is_best, time_str):

        # log details
        log_string = ("Epoch [{:^5}]: Train Avg accuracy: {:.4f}; Train Avg loss: {:.4f} \n".format(n_epoch, acc_train, loss_train) +
                      "{:14} Valid Avg accuracy: {:.4f}; Valid Avg loss: {:.4f} \n".format(" ", acc_val, loss_val) +
                      "{:14} Time: {}".format(" ", time_str) +
                      ("\n{:14} BEST MODEL SAVED".format(" ") if is_best else ""))
        self.log_global(log_string)

        self.f_log_epoch.write("{:6d} {:14.4f} {:14.4f} {:14.4f} {:14.4f} {:>14}\n".
                               format(n_epoch, acc_train, loss_train, acc_val, loss_val, time_str))
        self.f_log_epoch.flush()

        # make visdom logging
        if self.with_visdom:
            self.writer_vis.line(X=np.array([[n_epoch, n_epoch]]),
                                 Y=np.array([[loss_train, loss_val]]),
                                 opts=dict(legend=['train', 'test']),
                                 win=self.avg_loss_window, update='append', env=self.experiment_name)
            self.writer_vis.line(X=np.array([[n_epoch, n_epoch]]),
                                 Y=np.array([[acc_train, acc_val]]),
                                 opts=dict(legend=['train', 'test']),
                                 win=self.avg_acc_window, update='append', env=self.experiment_name)

        if self.with_tensorboard:
            self.writer_tb.add_scalar('Train/Loss', loss_train, n_epoch)
            self.writer_tb.add_scalar('Train/Accuracy', acc_train, n_epoch)
            self.writer_tb.add_scalar('Test/Loss', loss_val, n_epoch)
            self.writer_tb.add_scalar('Test/Accuracy', acc_val, n_epoch)

    def log_global(self, log_str):
        self.f_log_global.write(log_str + "\n")
        self.f_log_global.flush()
        print(log_str)

    def close(self):
        # close log files
        self.f_log_iter.close()
        self.f_log_epoch.close()
        self.f_log_global.close()

