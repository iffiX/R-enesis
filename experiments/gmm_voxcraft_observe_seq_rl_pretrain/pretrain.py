import os
import multiprocessing
import tqdm
import h5py
import numpy as np
import torch as t
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Any
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from renesis.env.virtual_shape import (
    VirtualShapeGMMObserveWithVoxelEnvironment,
    normalize,
)
from experiments.gmm_voxcraft_observe_seq_rl_pretrain.config import (
    pretrain_config,
    steps,
    dimension_size,
    materials,
)
from experiments.gmm_voxcraft_observe_seq_rl_pretrain.model import Actor


class PretrainDatasetGeneratorParallelContext:
    worker_id: int = None
    worker_data: Any = None


class PretrainDatasetGenerator:
    def __init__(
        self,
        dataset_path: str,
        steps: int,
        dimension_size: int,
        materials: Tuple[int],
        episode_num_for_train: int,
        episode_num_for_validate: int,
    ):
        self.steps = steps
        self.dimension_size = dimension_size
        self.materials = materials

        generate = False

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        if not os.path.exists(dataset_path):
            print(f"Dataset not found in: {dataset_path}, generating")
            generate = True
        else:
            try:
                with h5py.File(dataset_path, mode="r") as file:
                    if (
                        file.attrs["steps"] != steps
                        or file.attrs["dimension_size"] != dimension_size
                        or file.attrs["materials"] != str(materials)
                    ):
                        print(f"Dataset found but configuration mismatch, regenerating")
                        os.remove(dataset_path)
                        generate = True
            except OSError:
                print("Invalid dataset file, regenerating")
                os.remove(dataset_path)
                generate = True

        if generate:
            env = VirtualShapeGMMObserveWithVoxelEnvironment(
                config={
                    "materials": materials,
                    "max_gaussian_num": steps,
                    "dimension_size": dimension_size,
                    "reference_shape": np.zeros([dimension_size] * 3),
                    "max_steps": steps,
                    "reward_type": "none",
                }
            )

            observation_space = env.observation_space
            action_space = env.action_space
            expected_size = (
                (episode_num_for_train + episode_num_for_validate)
                * steps
                * (1 + action_space.shape[0] + observation_space.shape[0])
                * 4
                / 1024**2
            )
            print(f"Expected size: {expected_size:.2f} MiB")
            with h5py.File(dataset_path, mode="w", rdcc_nbytes=1024**3) as file:
                file.attrs["steps"] = steps
                file.attrs["dimension_size"] = dimension_size
                file.attrs["materials"] = str(materials)
                file.attrs["action_space_size"] = action_space.shape[0]
                file.attrs["observation_space_size"] = observation_space.shape[0]
                # Last dimension stores the step number, the action and the
                # observation after that action. The initial observation is
                # omitted since it's always zero.
                # The step number starts at 0, For episodes shorter than steps,
                # the step number will be -1 for uninitialized actions and
                # observations.
                for split, split_episode_num in (
                    ("train", episode_num_for_train),
                    ("validate", episode_num_for_validate),
                ):
                    dataset = file.create_dataset(
                        split,
                        shape=(
                            split_episode_num,
                            steps,
                            1 + action_space.shape[0] + observation_space.shape[0],
                        ),
                        chunks=(
                            100,
                            20,
                            1 + action_space.shape[0] + observation_space.shape[0],
                        ),
                        dtype=np.float32,
                    )
                    self.fill_data(dataset, split_episode_num)

    def fill_data(self, dataset, episode_num):
        for seed_chunk in self.chunk(
            list(range(episode_num)), chunk_size=1000 * multiprocessing.cpu_count()
        ):
            print(
                f"Generating data {min(seed_chunk)} to {max(seed_chunk)}, total {episode_num}"
            )
            # Should consume around
            results = np.stack(
                self.parallel_run(
                    seed_chunk,
                    self.fill_data_worker,
                    initargs=(self.steps, self.dimension_size, self.materials),
                    initializer=self.fill_data_worker_initializer,
                ),
            )
            dataset[seed_chunk] = results

    @staticmethod
    def fill_data_worker_initializer(steps, dimension_size, materials):
        env = VirtualShapeGMMObserveWithVoxelEnvironment(
            config={
                "materials": materials,
                "max_gaussian_num": steps,
                "dimension_size": dimension_size,
                "reference_shape": np.zeros([dimension_size] * 3),
                "max_steps": steps,
                "reward_type": "none",
            }
        )
        PretrainDatasetGeneratorParallelContext.worker_data = (
            env,
            steps,
            dimension_size,
            materials,
        )

    @staticmethod
    def fill_data_worker(seed):
        env, steps, *_ = PretrainDatasetGeneratorParallelContext.worker_data

        episode = np.zeros(
            [steps, 1 + env.action_space.shape[0] + env.observation_space.shape[0]]
        )
        episode[:, 0] = -1
        rand = np.random.RandomState(seed)
        env.reset()
        for step in range(steps):
            # TODO: find an appropriate way to generate this
            action = rand.rand(*env.action_space.shape) * 4 - 2
            obs, _, done, __ = env.step(action)
            episode[step] = np.concatenate(
                [np.array([step], dtype=np.float32), action, obs], axis=0
            )
            if done:
                break
        return episode

    @staticmethod
    def chunk(list_to_chunk: List[Any], chunk_size: int):
        return [
            list_to_chunk[start : min(len(list_to_chunk), start + chunk_size)]
            for start in range(0, len(list_to_chunk), chunk_size)
        ]

    def parallel_run(
        self,
        inputs,
        func,
        pesudo=False,
        starting_method="fork",
        processes=None,
        initializer=None,
        initargs=None,
    ):
        results = []
        if not pesudo:
            processes = processes or multiprocessing.cpu_count()
            chunk_size = max((len(inputs) + processes - 1) // processes, 1)
            ctx = multiprocessing.get_context(starting_method)
            queue = ctx.Queue()
            process_pool = [
                ctx.Process(
                    target=self.parallel_executor,
                    args=(
                        worker_id,
                        processes,
                        initializer,
                        initargs,
                        func,
                        chunk,
                        queue,
                        len(inputs),
                    ),
                )
                for worker_id, chunk in zip(
                    range(processes),
                    self.chunk(inputs, chunk_size=chunk_size),
                )
            ]

            for p in process_pool:
                p.start()
            for _ in process_pool:
                results.append(queue.get())
            for p in process_pool:
                p.join()
            results = sorted(results, key=lambda x: x[0])
            results = [xx for x in results for xx in x[1]]

        else:
            PretrainDatasetGeneratorParallelContext.worker_id = 0
            if initializer is not None:
                initargs = initargs or ()
                initializer(*initargs)
            for fact in tqdm.tqdm(inputs):
                result = func(fact)
                if result is not None:
                    results.append(result)
        return results

    @staticmethod
    def parallel_executor(
        worker_id, worker_num, initializer, initargs, func, split, queue, total_num
    ):
        PretrainDatasetGeneratorParallelContext.worker_id = worker_id
        if initializer is not None:
            initargs = initargs or ()
            initializer(*initargs)
        if worker_id == 0:
            with tqdm.tqdm(total=total_num) as bar:
                result = []
                for s in split:
                    res = func(s)
                    if res is not None:
                        result.append(res)
                    bar.update(worker_num)
                queue.put((worker_id, result))
        else:
            queue.put(
                (worker_id, [x for x in [func(s) for s in split] if x is not None])
            )


# class PretrainDataset(Dataset):
#     def __init__(self, dataset_path, split):
#         self.file = h5py.File(dataset_path, mode="r", rdcc_nbytes=1024**3)
#         self.dataset = self.file[split]
#
#         self.steps = self.file.attrs["steps"]
#         self.dimension_size = self.file.attrs["dimension_size"]
#         self.action_space_size = self.file.attrs["action_space_size"]
#         self.observation_space_size = self.file.attrs["observation_space_size"]
#
#         self.index = []
#         step_num = self.dataset[:, :, 0]
#         for episode in range(self.dataset.shape[0]):
#             for step in range(self.dataset.shape[1]):
#                 if step_num[episode, step] != -1:
#                     self.index.append((episode, step))
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, item):
#         episode, step = self.index[item]
#         all_past_obs = np.zeros(
#             [self.steps, self.observation_space_size], dtype=np.float32
#         )
#         predict_voxels = np.zeros(
#             [self.steps, self.dimension_size**3], dtype=np.float32
#         )
#         if step > 0:
#             all_past_obs[-step:] = self.dataset[
#                 episode, :step, 1 + self.action_space_size :
#             ]
#         predict_voxels[-(step + 1) :] = self.dataset[
#             episode, : step + 1, -self.dimension_size**3 :
#         ]
#         return all_past_obs, predict_voxels


# class PretrainDataset(Dataset):
#     def __init__(self, dataset_path, split):
#         with h5py.File(dataset_path, mode="r") as file:
#             self.dataset = file[split][:]
#
#             self.steps = file.attrs["steps"]
#             self.dimension_size = file.attrs["dimension_size"]
#             self.action_space_size = file.attrs["action_space_size"]
#             self.observation_space_size = file.attrs["observation_space_size"]
#
#             self.index = []
#             step_num = self.dataset[:, :, 0]
#             for episode in range(self.dataset.shape[0]):
#                 for step in range(self.dataset.shape[1]):
#                     if step_num[episode, step] != -1:
#                         self.index.append((episode, step))
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, item):
#         episode, step = self.index[item]
#         all_past_obs = np.zeros(
#             [self.steps, self.observation_space_size], dtype=np.float32
#         )
#         predict_voxels = np.zeros(
#             [self.steps, self.dimension_size**3], dtype=np.float32
#         )
#         if step > 0:
#             all_past_obs[-step:] = self.dataset[
#                 episode, :step, 1 + self.action_space_size :
#             ]
#         predict_voxels[-(step + 1) :] = self.dataset[
#             episode, : step + 1, -self.dimension_size**3 :
#         ]
#         return all_past_obs, predict_voxels


class PretrainDataset(Dataset):
    def __init__(self, dataset_path, split):
        with h5py.File(dataset_path, mode="r") as file:
            print(f"Loading {split} dataset to GPU")
            self.dataset = t.from_numpy(file[split][:]).to(device="cuda:0")
            print("Loaded")
            self.steps = file.attrs["steps"]
            self.dimension_size = file.attrs["dimension_size"]
            self.action_space_size = file.attrs["action_space_size"]
            self.observation_space_size = file.attrs["observation_space_size"]

            self.index = []
            step_num = self.dataset[:, :, 0]
            for episode in range(self.dataset.shape[0]):
                for step in range(self.dataset.shape[1]):
                    if step_num[episode, step] != -1:
                        self.index.append((episode, step))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode, step = self.index[item]
        all_past_obs = t.zeros(
            [self.steps, self.observation_space_size], dtype=t.float32, device="cuda:0"
        )
        predict_voxels = t.zeros(
            [self.steps, self.dimension_size**3], dtype=t.float32, device="cuda:0"
        )
        if step > 0:
            all_past_obs[-step:] = self.dataset[
                episode, :step, 1 + self.action_space_size :
            ]
        predict_voxels[-(step + 1) :] = self.dataset[
            episode, : step + 1, -self.dimension_size**3 :
        ]
        return all_past_obs, predict_voxels


class Pretrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.env = VirtualShapeGMMObserveWithVoxelEnvironment(config["env_config"])
        self.model = Actor(
            self.env.observation_space,
            self.env.action_space,
            self.env.action_space.shape[0] * 2,
            config["model"],
            "actor",
            **config["model"]["custom_model_config"],
        )
        self.dataset_path = config["dataset_path"]
        self.seed = config["seed"]
        self.dataloader_args = config["dataloader_args"]

    @property
    def monitor(self):
        return "f1"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        gen = t.Generator()
        gen.manual_seed(self.seed)
        dataset = PretrainDataset(self.dataset_path, "train")
        return DataLoader(
            dataset=dataset,
            sampler=RandomSampler(dataset, generator=gen),
            **self.dataloader_args,
        )

    def val_dataloader(self):
        dataset = PretrainDataset(self.dataset_path, "validate")
        return DataLoader(
            dataset=dataset,
            **self.dataloader_args,
        )

    def training_step(self, batch, batch_idx):
        all_past_obs, predict_voxels = batch
        predicted_voxels_one_hot_mean, *_ = self.model(
            {
                "obs": all_past_obs[:, -1].to(
                    self.device
                ),  # dummy input, make ray model not complain
                "custom_obs": all_past_obs.to(self.device),
                "return_voxel": True,
            },
            None,
            None,
        )
        loss = F.mse_loss(
            predicted_voxels_one_hot_mean,
            self.model.to_one_hot_voxels(predict_voxels).to(self.device),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        all_past_obs, predict_voxels = batch
        predict_voxels = predict_voxels[:, -1]
        predicted_voxels_one_hot_mean, *_ = self.model(
            {
                "obs": all_past_obs[:, -1].to(
                    self.device
                ),  # dummy input, make ray model not complain
                "custom_obs": all_past_obs.to(self.device),
                "return_voxel": True,
            },
            None,
            None,
        )
        # Only compare the last predicted voxels
        last_predicted_voxels = t.argmax(
            predicted_voxels_one_hot_mean[:, -1].reshape(
                predicted_voxels_one_hot_mean.shape[0], len(self.model.materials), -1
            ),
            dim=1,
        )
        all_scores = []
        for sample_idx in range(len(last_predicted_voxels)):
            occurences = [
                int(t.sum(predict_voxels[sample_idx] == mat))
                for mat in self.model.materials
            ]
            # Test notes:
            # use 1/occurrence is too small for voxels with large quantities
            weights = np.array(
                [
                    1 / np.log(occurrence) if occurrence > 1 else 0
                    for occurrence in occurences
                ]
            )
            scores = f1_score(
                predict_voxels[sample_idx].cpu().numpy(),
                last_predicted_voxels[sample_idx].cpu().numpy(),
                labels=self.model.materials,
                average=None,
                zero_division=0,
            )
            all_scores.append(np.average(scores, weights=weights))
        self.log("f1", np.mean(all_scores), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.config["lr"])


if __name__ == "__main__":
    gen = PretrainDatasetGenerator(
        pretrain_config["dataset_path"],
        steps=steps,
        dimension_size=dimension_size,
        materials=materials,
        episode_num_for_train=pretrain_config["episode_num_for_train"],
        episode_num_for_validate=pretrain_config["episode_num_for_validate"],
    )
    pretrainer = Pretrainer(pretrain_config)
    os.makedirs(pretrain_config["checkpoint_path"], exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=pretrain_config["checkpoint_path"],
        filename="{epoch:02d}-"
        + pretrainer.monitor
        + "-{"
        + pretrainer.monitor
        + ":.3f}",
        save_top_k=20,
        monitor=pretrainer.monitor,
        mode=pretrainer.monitor_mode,
        verbose=True,
    )
    os.makedirs(pretrain_config["log_path"], exist_ok=True)
    t_logger = TensorBoardLogger(save_dir=pretrain_config["log_path"])
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=[checkpoint_callback],
        logger=[t_logger],
        max_epochs=pretrain_config["epochs"],
        deterministic=True,
        precision=32,
    )
    trainer.stage_mode = "train"
    trainer.fit(pretrainer)
    pretrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    t.save(pretrainer.model.state_dict(), "~/data/renesis/pretrain/result/model.pt")
