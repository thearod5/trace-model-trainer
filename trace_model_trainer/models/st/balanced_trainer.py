from sentence_transformers import SentenceTransformerTrainer
from torch.utils.data import DataLoader

from trace_model_trainer.models.st.balanced_data_loader import BalancedSampler


class BalancedTrainer(SentenceTransformerTrainer):
    def get_train_dataloader(self) -> DataLoader:
        data_loader = super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            collate_fn=data_loader.collate_fn,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            persistent_workers=data_loader.persistent_workers,
            prefetch_factor=data_loader.prefetch_factor,
            batch_sampler=BalancedSampler(self.train_dataset),
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            pin_memory_device=data_loader.pin_memory_device
            # batch_size, shuffle, sampler, drop_last defined in sampler
        )
