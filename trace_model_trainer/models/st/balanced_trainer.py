from sentence_transformers import SentenceTransformerTrainer
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from trace_model_trainer.models.st.balanced_sampler import BalancedSampler
from trace_model_trainer.transforms.augmentation import create_augmented_dataset


class BalancedTrainer(SentenceTransformerTrainer):

    def get_train_dataloader(self) -> DataLoader:
        data_loader = super().get_train_dataloader()

        return DataLoader(
            data_loader.dataset,
            collate_fn=data_loader.collate_fn,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            persistent_workers=data_loader.persistent_workers,
            prefetch_factor=data_loader.prefetch_factor,
            batch_sampler=BalancedSampler(data_loader.dataset, batch_size=self.args.per_device_train_batch_size),
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            pin_memory_device=data_loader.pin_memory_device
            # batch_size, shuffle, sampler, drop_last defined in sampler
        )


class DataAugmentationCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Regenerate the augmented dataset at the beginning of each epoch
        original_dataset = self.trainer.original_dataset
        self.trainer.train_dataset = {k: create_augmented_dataset(original_dataset[k])
                                      for k in original_dataset.keys()}


class AugmentedTrainer(BalancedTrainer):
    def __init__(self, *args, **kwargs):
        self.original_dataset = kwargs["train_dataset"]
        kwargs["train_dataset"] = {k: create_augmented_dataset(self.original_dataset[k]) for k in
                                   self.original_dataset.keys()}  # needed to ensure columns are recognized later on
        super().__init__(*args, **kwargs)
        self.add_callback(DataAugmentationCallback(self))
