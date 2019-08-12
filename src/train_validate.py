import torch

from thirdparty.meters import AverageMeter, accuracy


def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, device, log):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    # number of training samples
    num_iter = len(train_loader)

    for i, (input_data, target) in enumerate(train_loader):
        target = target.to(device)
        input_data = input_data.to(device)

        # compute output
        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss, input_data.size(0))
        top1.update(prec1[0], input_data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        time_string = log.timers['global'].current2str()  # get current time
        log.log_iter(i, epoch, num_iter, losses.val.to('cpu').item(), time_string)

    # adjust_learning_rate
    if scheduler is not None:
        scheduler.step()

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()


def validate(val_loader, model, criterion, device):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            target = target.to(device)
            input_data = input_data.to(device)

            # compute output
            output = model(input_data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss, input_data.size(0))
            top1.update(prec1[0], input_data.size(0))

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()
