import logging
import time

class Trainer(object):
    def __init__(self, cfg, model, rank, output_dir, writer_dict):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ

    def train(self, epoch, data_loader, optimizer):
        logger = logging.getLogger("Training")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        multi_heatmap_loss_meter = AverageMeter()
        single_heatmap_loss_meter = AverageMeter()
        contrastive_loss_meter = AverageMeter()

        self.model.train()

        end = time.time()
        for i, batched_inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            loss_dict = self.model(batched_inputs)

            loss = 0
            num_images = len(batched_inputs)
            if 'multi_heatmap_loss' in loss_dict:
                multi_heatmap_loss = loss_dict['multi_heatmap_loss']
                multi_heatmap_loss_meter.update(multi_heatmap_loss.item(), num_images)
                loss += multi_heatmap_loss

            if 'single_heatmap_loss' in loss_dict:
                single_heatmap_loss = loss_dict['single_heatmap_loss']
                single_heatmap_loss_meter.update(single_heatmap_loss.item(), num_images)
                loss += single_heatmap_loss
            
            if 'contrastive_loss' in loss_dict:
                contrastive_loss = loss_dict['contrastive_loss']
                contrastive_loss_meter.update(contrastive_loss.item(), num_images)
                loss += contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 and self.rank == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{multiple}{single}{contrast}'.format(
                        epoch, i, len(data_loader),
                        batch_time=batch_time,
                        speed=num_images / batch_time.val,
                        data_time=data_time,
                        multiple=_get_loss_info(multi_heatmap_loss_meter, 'multiple'),
                        single=_get_loss_info(single_heatmap_loss_meter, 'single'),
                        contrast=_get_loss_info(contrastive_loss_meter, 'contrast')
                    )
                logger.info(msg)

def _get_loss_info(meter, loss_name):
    msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
    return msg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0