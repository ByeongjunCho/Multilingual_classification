import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import transformers
import pytorch_lightning as pl
from sklearn.metrics import classification_report # for logging performance measures
import torchmetrics

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from datasets import load_dataset

import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class KOTEEkmanDataset(Dataset):
    def __init__(self, encodings, labels):
        self.labels = labels
        self.encodings = encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        encode_key = self.encodings.keys()
        mydict = {k: self.encodings[k][idx] for k in encode_key}
        mydict['labels'] = torch.FloatTensor(self.labels[idx])

        return mydict

class RoBERTaLitModel(pl.LightningModule):
    def __init__(self, model, tokenizer, opt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.opt = opt
        self.lr = opt.lr

        # self.n_training_steps = n_training_steps
        # self.n_warmup_steps = n_warmup_steps

        ## the loss
        # self.criterion = criterion

    def forward(self, x):
        output = self.model(**x)
        # cls_tokens = output[0][:,0,:]
        # output = self.classifier(cls_tokens)

        # torch.cuda.empty_cache()

        return output

    def step(self, x):
        outputs = self.forward(x)
        loss = outputs.loss
        logits = torch.sigmoid(outputs.logits)
        logits = torch.where(logits>0.5, 1, 0) # 0 or 1로 구성된 값으로 변경

        # metrics 계산(F1 score)
        macro_f1 = torchmetrics.functional.classification.multilabel_f1_score(logits, x['labels'], num_labels=logits.size(-1), average='macro')
        micro_f1 = torchmetrics.functional.classification.multilabel_f1_score(logits, x['labels'], num_labels=logits.size(-1), average='micro')
        weighted_f1 = torchmetrics.functional.classification.multilabel_f1_score(logits, x['labels'], num_labels=logits.size(-1), average='weighted')

        # accuracy
        acc = (logits == x['labels']).sum() / (x['labels'].size(0) * x['labels'].size(1))

        return {"outputs": outputs,
                "loss": loss,
                "logits": logits, # consisted 0 or 1(passing sigmoid and 0 or 1 processed)
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "weighted_f1": weighted_f1,
                "acc": acc
                }

    def training_step(self, batch, batch_idx):
        mode="train"
        outputs = self.step(batch)

        self.log(f'{mode}_loss', outputs['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_acc', outputs['acc'], on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_macro_f1', outputs['macro_f1'], on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_micro_f1', outputs['micro_f1'], on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_weighted_f1', outputs['weighted_f1'], on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict({f'{mode}_macro_f1' : outputs['macro_f1'],
                       f'{mode}_micro_f1' : outputs['micro_f1'],
                       f'{mode}_weighted_f1' : outputs['weighted_f1']},
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True)

        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_idx):
        mode="val"
        outputs = self.step(batch)

        self.log(f'{mode}_loss', outputs['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_acc', outputs['acc'], on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f'{mode}_macro_f1', outputs['macro_f1'], on_step=True, on_epoch=True, sync_dist=True)
        # self.log(f'{mode}_micro_f1', outputs['micro_f1'], on_step=True, on_epoch=True, sync_dist=True)
        # self.log(f'{mode}_weighted_f1', outputs['weighted_f1'], on_step=True, on_epoch=True, sync_dist=True)

        return {'y_pred': outputs['logits'], 'y_true': batch['labels']}

    def test_step(self, batch, batch_idx):
        mode = "test"
        outputs = self.step(batch)

        # self.log(f'{mode}_loss', outputs['loss'], on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f'{mode}_acc', outputs['acc'], on_step=False, on_epoch=True, sync_dist=True)

        return {'y_pred': outputs['logits'], 'y_true': batch['labels']}

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # real train step calculate(consider gpus)
        n_gpus = self.trainer.num_gpus
        train_step = (len(self.train_dataloader()) // n_gpus) + 1 if len(self.train_dataloader()) % n_gpus else len(self.train_dataloader()) // n_gpus
        scheduler = transformers.get_scheduler(name='linear',
                                               optimizer=optimizer,
                                               num_warmup_steps=(self.opt.n_epochs*train_step) // 5,
                                               num_training_steps=self.opt.n_epochs*train_step
                                               )
        # scheduler = transformers.get_scheduler(name='linear',
        #                                        optimizer=optimizer,
        #                                        num_warmup_steps=(self.opt.n_epochs * self.opt.train_step) // 5,
        #                                        num_training_steps=self.opt.n_epochs * self.opt.train_step
        #
        #                                        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval="step"
            )
        )
        # return dict(
        #     optimizer=optimizer
        # )

    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)  # outputs = [local rank 0, local rank 1]
        # local_rank 0 = {'val_loss': torch.tensor([y_pred_step0, y_pred_step1, ....])
        # if self.trainer.is_global_zero == 0: # only 1 process running
        if self.local_rank == 0:  # only 1 process running
            y_true, y_pred = [], []
            for out_dict in outputs:
                y_true.append(torch.cat([x for x in out_dict['y_true']]))
                y_pred.append(torch.cat([x for x in out_dict['y_pred']]))
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

            y_true = y_true.detach().cpu().tolist()
            y_pred = y_pred.detach().cpu().tolist()

            classes = ['anger',
                       'disgust',
                       'fear',
                       'joy',
                       'sadness',
                       'surprise',
                       'neutral']
            report = classification_report(y_true, y_pred, output_dict=True,
                                           target_names=classes)


            # logging all things
            for key in report.keys():
                mydict = {"valid_"+key + "_" + k: torch.tensor(v, dtype=torch.float32) for k, v in report[key].items()}
                self.log_dict(mydict, prog_bar=False, rank_zero_only=True)

    def test_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)  # outputs = [local rank 0, local rank 1]
        # local_rank 0 = {'val_loss': torch.tensor([y_pred_step0, y_pred_step1, ....])
        # if self.trainer.is_global_zero == 0: # only 1 process running
        if self.local_rank == 0:  # only 1 process running
            y_true, y_pred = [], []
            for out_dict in outputs:
                y_true.append(torch.cat([x for x in out_dict['y_true']]))
                y_pred.append(torch.cat([x for x in out_dict['y_pred']]))
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)

            y_true = y_true.detach().cpu().tolist()
            y_pred = y_pred.detach().cpu().tolist()

            classes = ['anger',
                       'disgust',
                       'fear',
                       'joy',
                       'sadness',
                       'surprise',
                       'neutral']
            report = classification_report(y_true, y_pred, output_dict=True,
                                           target_names=classes)
            print(report)

    def prepare_data(self) -> None:
        # load goemotions datasets
        kote_ekman = load_dataset("kjhkjh95/kote_ekman")

        # goemotions label mapping dict
        # go_label2idx = {v: i for i, v in enumerate(goemotions['train'].features['labels'].feature.names)}
        # go_idx2label = {i: v for i, v in enumerate(goemotions['train'].features['labels'].feature.names)}

        # ekman mapping dict
        ekman_mapping = {'anger': ['anger', 'annoyance', 'disapproval'],
                         'disgust': ['disgust'],
                         'fear': ['fear', 'nervousness'],
                         'joy': ['joy',
                                 'amusement',
                                 'approval',
                                 'excitement',
                                 'gratitude',
                                 'love',
                                 'optimism',
                                 'relief',
                                 'pride',
                                 'admiration',
                                 'desire',
                                 'caring'],
                         'sadness': ['sadness', 'disappointment', 'embarrassment', 'grief', 'remorse'],
                         'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
                         'neutral': ['neutral', 'no emotion']}

        # tokenize
        self.train_encodings = self.tokenizer(kote_ekman['train']['text'],
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors="pt",
                                             )
        self.val_encodings = self.tokenizer(kote_ekman['validation']['text'],
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors="pt",
                                             )
        self.test_encodings = self.tokenizer(kote_ekman['test']['text'],
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors="pt",
                                             )

        # label ekman mapping and transform multilabel task
        ekman_label2idx = {'anger': 0,
                        'disgust': 1,
                        'fear': 2,
                        'joy': 3,
                        'sadness': 4,
                        'surprise': 5,
                        'neutral': 6}

        ekman_idx2label = {v:k for k, v in ekman_label2idx.items()}
        # 1. ekamn mapping
        label_dict = {}
        for m in ['train', 'validation', 'test']:
            tmp = []
            labels = kote_ekman[m]['labels']

            for label in labels:
                ekman_ = set()
                for l in eval(label): # multilabel
                    if l == "No Emotion": # change no_emotion to neutral
                        l = "neutral"
                    ekman_.add(ekman_label2idx[l.lower()])
                tmp.append(list(ekman_))

            label_dict[m] = tmp

        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()

        self.train_labels = mlb.fit_transform(label_dict["train"])
        self.test_labels = mlb.fit_transform(label_dict["test"])
        self.val_labels = mlb.fit_transform(label_dict["validation"])

    def setup(self, stage):
        if stage == 'fit':
            self.train_ds = KOTEEkmanDataset(self.train_encodings, self.train_labels)
            self.val_ds = KOTEEkmanDataset(self.val_encodings, self.val_labels)

        if stage == 'test':
            self.test_ds = KOTEEkmanDataset(self.test_encodings, self.test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=8,
                          batch_size=self.opt.batch_size
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=8,
                          batch_size=self.opt.batch_size
                          )

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=8,
                          batch_size=self.opt.batch_size
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")

    # WandbLogger
    parser.add_argument('--proj_name', type=str, default='Multilingual', help='wandb logger project name')
    parser.add_argument('--model_name', type=str, default='kote1', help='wandb logger model name')
    parser.add_argument('--group_name', type=str, default='XLMRoBERTa', help='wandb logger group name')
    parser.add_argument('--version', type=str, default=None, help='wandb version')

    # pytorch lightning Trainer option
    parser.add_argument('--precision', type=int, default=32, help="precision learning [16 | 32]")
    parser.add_argument('--accelerator', type=str, default='gpu', help="accelerator for training [cpu | gpu]")
    parser.add_argument('--resume', type=bool, default=False, help='resume training marker [True | False]')
    parser.add_argument('--devices', type=int, nargs='+', default=-1)

    # etc
    parser.add_argument('--seed', type=int, default=59, help="random seed On/Off")

    opt = parser.parse_args()
    if opt.seed:
        pl.seed_everything(opt.seed)

    # load model and tokenizer

    model = transformers.AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=7,
                                                                            problem_type="multi_label_classification")
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")

    # load pretrained model
    plmodel = RoBERTaLitModel(model, tokenizer, opt)
    plmodel.load_from_checkpoint(
        "./Multilingual/goemotions1/ckpt/epoch=00004-val_loss=0.21969-val_acc=0.909.model.ckpt",
        model=model, tokenizer=tokenizer, opt=opt)


    # logging path root
    save_root_path = f"{opt.proj_name}/{opt.model_name}/"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch:05d}-{val_loss:.5f}-{val_acc:.3f}.model',
        save_top_k=-1,
        save_last=True,
        save_weights_only=True,
        dirpath=save_root_path + '/ckpt',
        every_n_epochs=1
    )

    # learningrate callback
    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=True
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=300,
        verbose=False,
        mode='min')

    # wandb logger
    wandb_logger = WandbLogger(project=opt.proj_name,
                               name=opt.model_name,
                               # job_type='train',
                               config=opt,
                               log_model=False,
                               version=opt.version if opt.version else None,
                               save_dir=save_root_path,
                               group=opt.group_name if opt.group_name else None
                               )
    wandb_logger.watch(model, 'all')

    # tensorboard logger
    tensorboard_logger = TensorBoardLogger(save_dir=save_root_path + "/tb_logs",
                                           name=opt.model_name,
                                           log_graph=True,
                                           )

    trainer = Trainer(
        max_epochs=opt.n_epochs,
        precision=opt.precision,
        accelerator=opt.accelerator,
        callbacks=[checkpoint_callback, lr_callback],
        logger=[wandb_logger, tensorboard_logger],
        # logger=wandb_logger,
        log_every_n_steps=1,
        devices=opt.devices,
        # resume_from_checkpoint=save_root_path + 'model_ckpt/last.ckpt' if opt.resume else None,
        strategy='ddp',
        # for debugging
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # num_sanity_val_steps=0
    )

    # trainer = Trainer(
    #     max_epochs=opt.n_epochs,
    #     precision=opt.precision,
    #     accelerator=opt.accelerator,
    #     log_every_n_steps=1,
    #     devices=opt.devices,
    #     # resume_from_checkpoint=save_root_path + 'model_ckpt/last.ckpt' if opt.resume else None,
    #     strategy='ddp',
    #     # for debugging
    #     limit_train_batches=0.1,
    #     limit_val_batches=0.1,
    # )

    plmodel.prepare_data()
    trainer.fit(plmodel)