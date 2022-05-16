import argparse
import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from utill.util import enumerateWithEstimate
from combining_data.combining_data import LunaDataset
from .model import LunaModel

from utill.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):

        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )

        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )        

        parser.add_argument('--tb-prefix',
            default='classification',
            help="Data prefix to use for Tensorboard run.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='lcd-pt',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.train_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA. {} devices".format(torch.cuda.device_count()))

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model = model.to(self.device)
        return model


    def initOptimizer(self):

        return SGD(self.model.parameters(), lr = 0.001, momentum=0.99)
        # return Adam(self.model.parameters())


    def initTrainDL(self):
        train_ds = LunaDataset(
            val_step=10,
            isValset=False,
            ratio_int= int(self.cli_args.balanced),
            augmen_dict=self.augmentation_dict,
            )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return train_dl


    def initValDL(self):
        val_ds = LunaDataset(
            val_step=10,
            isValset=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size = batch_size,
            num_workers= self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return val_dl


    def initTensorboardWriters(self):
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDL()
        val_dl = self.initValDL()

        for epoch_index in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_index,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trainMetrics_tensor = self.doTraining(epoch_index, train_dl)
            self.logMetrics(epoch_index, 'train', trainMetrics_tensor)

            valMetrics_tensor = self.doValidation(epoch_index, val_dl)
            self.logMetrics(epoch_index, 'val', valMetrics_tensor)


        if  hasattr(self, 'train_writer'):
            self.train_writer.close()
            self.val_writer.close()


    def doTraining(self, epoch_index, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()

        trainMetrics_gpu = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device = self.device
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} training".format(epoch_index),
            start_index= train_dl.num_workers
        )

        for batch_index, batch_tuple in batch_iter:
            self.optimizer.zero_grad()

            loss = self.computeBatchLoss(
                batch_index,
                batch_tuple,
                train_dl.batch_size,
                trainMetrics_gpu
            )

            loss.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.train_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.train_writer.close()


        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trainMetrics_gpu.to('cpu')


    def doValidation(self, epoch_index, val_dl):

        with torch.no_grad():
            self.model.eval()
            valMetrics_gpu = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device = self.device
            )
            
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_index),
                start_index = val_dl.num_workers
            )

            for batch_index, batch_tuple in batch_iter:
                self.computeBatchLoss(
                    batch_index,
                    batch_tuple,
                    val_dl.batch_size,
                    valMetrics_gpu
                )

            return valMetrics_gpu.to('cpu')

    
    def computeBatchLoss(self,
        batch_index,
        batch_tuple,
        batch_size, 
        metrics_gpu
    ):

        input_tensor, label_tensor, _series_list, _center_list = batch_tuple

        input_gpu = input_tensor.to(self.device, non_blocking = True)
        label_gpu = label_tensor.to(self.device, non_blocking = True)

        logits_gpu, probability_gpu = self.model(input_gpu)

        loss_fn = nn.CrossEntropyLoss(reduction = 'none')
        loss_gpu = loss_fn(
            logits_gpu,
            label_gpu[:,1]
            )

        start_index = batch_index * batch_size
        end_index = start_index + label_tensor.size(0)

        metrics_gpu[METRICS_LABEL_NDX, start_index:end_index] = label_gpu[:,1].detach()

        metrics_gpu[METRICS_PRED_NDX, start_index:end_index] = probability_gpu[:,1].detach()

        metrics_gpu[METRICS_LOSS_NDX, start_index:end_index] = loss_gpu.detach()

        return loss_gpu.mean()


    def logMetrics(self,
        epoch_index,
        mode_str, 
        metrics_tensor, 
        classThreshold = 0.5
    ):
        
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_index,
            type(self).__name__,
        ))

        negLabel_mask = metrics_tensor[METRICS_LABEL_NDX] <= classThreshold
        negPred_mask = metrics_tensor[METRICS_PRED_NDX] <= classThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_tensor[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_tensor[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_tensor[METRICS_LOSS_NDX, posLabel_mask].mean()

        
        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_tensor.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100


        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_index,
                mode_str,
                **metrics_dict,
            )
        )
        
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_index,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )

        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_index,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_tensor[METRICS_LABEL_NDX],
            metrics_tensor[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_tensor[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_tensor[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_tensor[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_tensor[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise



if __name__ == '__main__':
    LunaTrainingApp().main()