from typing import Any, List

import torch
import pytorch_lightning as pl

from torchmetrics import MaxMetric
from torchmetrics.retrieval.mean_reciprocal_rank import RetrievalMRR
from sentence_transformers.util import semantic_search

from src.models.components.loss import AdaptiveCriterion
from src.models.components.metrics import reciprocal_ranks
from src.utils import freeze_model, print_gpu_usage


class CrossModalLanguageModel(pl.LightningModule):
    """
    Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: AdaptiveCriterion,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        factor: float = 0.1,
        patience: int = 5,
        threshold: float = 0.001,
        cooldown: int = 0,
        trainable_text_layers: int = 1,
        trainable_speech_layers: int  = -1,
        trainable_multimodal_layers: int = -1
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False) #! appears to contain a bug
        
        # optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        
        #scheduler params
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown

        # model is instantiated by Hydra
        self.model = model
        # freeze_model(
        #     self.model, 
        #     trainable_text_layers=trainable_text_layers,
        #     trainable_speech_layers=trainable_speech_layers,
        #     trainable_multimodal_layers=trainable_multimodal_layers
        # )

        # loss function
        self.criterion = criterion

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mrr = RetrievalMRR(empty_target_action='neg')
        self.val_mrr = RetrievalMRR(empty_target_action='neg')
        self.test_mrr = RetrievalMRR(empty_target_action='neg')

        # for logging best so far validation accuracy
        self.val_mrr_best = MaxMetric()


    def forward(self, speech_inputs: torch.Tensor, text_inputs: torch.Tensor):
        return self.model(speech_inputs, text_inputs)


    def step(self, batch: Any):
        speech_inputs, text_inputs = batch
        
        # forward pass
        outputs = self.forward(speech_inputs, text_inputs)
        
        # compute loss
        loss = self.criterion(outputs)
        
        # compute reciprocal ranks
        pairwise_similarity = semantic_search(
            query_embeddings=outputs.text_pooler_output, 
            corpus_embeddings=outputs.speech_pooler_output,
            top_k=len(outputs.text_pooler_output))
        indexes, targets, preds = reciprocal_ranks(pairwise_similarity)
        
        return outputs, loss, indexes, targets, preds


    def training_step(self, batch: Any, batch_idx: int):
        outputs, loss, indexes, targets, preds = self.step(batch)
        
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # compute train metrics
        mrr = self.train_mrr(preds, targets, indexes)
        self.log("train/mrr", mrr, on_step=False, on_epoch=True, prog_bar=True)
        # self.train_mrr.update(mrr)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "mrr": mrr}


    def training_epoch_end(self, outputs: List[Any]):
        # self.log("train/mrr", self.train_mrr.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_mrr.reset()


    def validation_step(self, batch: Any, batch_idx: int):
        outputs, loss, indexes, targets, preds = self.step(batch)
        
        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # compute val metrics
        mrr = self.val_mrr(preds, targets, indexes)
        self.log("val/mrr", mrr, on_step=False, on_epoch=True, prog_bar=True)
        # self.val_mrr.update(mrr) #! <-- probably wrong, see: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls

        return {"loss": loss, "mrr": mrr}


    def validation_epoch_end(self, outputs: List[Any]):
        # self.log("val/mrr", self.val_mrr.compute(), on_step=False, on_epoch=True, prog_bar=True)
        
        # mrr = self.val_mrr.compute()  # get val mrr from current epoch
        # self.val_mrr_best.update(mrr)
        self.log("val/mrr_best", self.val_mrr_best.compute(), on_epoch=True, prog_bar=True)
        
        self.val_mrr.reset()
        

    def test_step(self, batch: Any, batch_idx: int):
        outputs, loss, indexes, targets, preds = self.step(batch)
        
        self.log("test/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # compute test metrics
        mrr = self.test_mrr(preds, targets, indexes)
        self.log("test/mrr", mrr, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_mrr.update(mrr)

        return {"loss": loss, "mrr": mrr}


    def test_epoch_end(self, outputs: List[Any]):
        # self.log("test/mrr", self.test_mrr.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.test_mrr.reset()


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience, threshold=self.threshold, cooldown=self.cooldown
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "train/loss"
            }
        }