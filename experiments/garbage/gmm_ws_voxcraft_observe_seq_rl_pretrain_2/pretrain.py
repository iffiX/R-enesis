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
from pytorch_lightning import seed_everything
from sklearn.metrics import f1_score
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from renesis.env.virtual_shape import (
    VirtualShapeGMMWSObserveWithVoxelAndRemainingStepsEnvironment,
    normalize,
)
from experiments.gmm_ws_voxcraft_observe_seq_rl_pretrain.config import (
    pretrain_config,
    steps,
    dimension_size,
    materials,
    sigma,
)
from experiments.gmm_ws_voxcraft_observe_seq_rl_pretrain.model import Actor


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
            env = VirtualShapeGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
                config=pretrain_config["env_config"]
            )

            observation_space = env.observation_space
            action_space = env.action_space
            expected_size = (
                (episode_num_for_train + episode_num_for_validate)
                * steps
                * observation_space.shape[0]
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
                # voxels before that action.
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
                            1 + steps,
                            action_space.shape[0] + observation_space.shape[0],
                        ),
                        chunks=(
                            100,
                            1 + steps,
                            action_space.shape[0] + observation_space.shape[0],
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
        env = VirtualShapeGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
            config=pretrain_config["env_config"]
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
        act_s, obs_s = env.action_space.shape[0], env.observation_space.shape[0]
        episode = np.zeros([1 + steps, act_s + obs_s])
        # Set first dimension non-initialized step records to -1
        episode[:, 0] = -1
        rand = np.random.RandomState(seed)
        obs = env.reset()
        episode[0, :] = 0
        episode[0, act_s:] = obs
        for step in range(steps):
            # TODO: find an appropriate way to generate this
            action = rand.rand(*env.action_space.shape) * 4 - 2
            obs, _, done, __ = env.step(action)
            episode[step + 1] = np.concatenate((action, obs))
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
                for step in range(self.dataset.shape[1] - 1):
                    if (
                        step_num[episode, step] != -1
                        and step_num[episode, step + 1] != -1
                    ):
                        self.index.append((episode, step))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode, step = self.index[item]
        past_obs = t.zeros(
            [self.steps, self.observation_space_size], dtype=t.float32, device="cuda:0"
        )
        predict_actions = t.zeros(
            [self.steps, self.action_space_size], dtype=t.float32, device="cuda:0"
        )
        predict_voxels = t.full(
            [self.steps, self.dimension_size**3],
            fill_value=-100,
            dtype=t.long,
            device="cuda:0",
        )
        # For past_obs, pad it to the left to make it consistent with the
        # ray trajectory view output
        # i.e. Suppose steps = 10, t=3, (t is step, t starts at 0) the observation is like:
        # [0, 0, 0, 0, 0, 0, obs_0, obs_1, obs_2, obs_3]
        # and obs_0 = 0
        past_obs[-(step + 1) :] = self.dataset[
            episode, : step + 1, self.action_space_size :
        ]

        # Since input is reordered in the model internally as:
        # [obs_0, obs_1, obs_2, obs_3, 0, 0, 0, 0, 0, 0]
        # Prediction target is
        # [obs_1, obs_2, obs_3, obs_4, -100, -100, -100, -100, -100, -100]
        predict_actions[:step] = self.dataset[
            episode, 1 : step + 1, : self.action_space_size
        ]
        predict_voxels[: step + 1] = self.dataset[
            episode, 1 : step + 2, -self.dimension_size**3 :
        ]
        assert self.dataset[episode, step + 1, 0] != -1
        return past_obs, predict_actions, predict_voxels, step


class Pretrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.env = VirtualShapeGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
            config["env_config"]
        )
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
        past_obs, predict_actions, predict_voxels, time = batch
        (predicted_actions, predicted_voxels_logits), *_ = self.model(
            {
                "obs": past_obs[:, -1].to(
                    self.device
                ),  # dummy input, make ray model not complain
                "custom_obs": past_obs.to(self.device),
                "return_voxel": True,
            },
            None,
            None,
        )
        # voxel_loss = F.cross_entropy(
        #     predicted_voxels_logits.view(
        #         (
        #             predict_voxels.shape[0],
        #             predict_voxels.shape[1],
        #             -1,
        #             predict_voxels.shape[2],
        #         )
        #     ).transpose(1, 2),
        #     predict_voxels,
        #     ignore_index=-100,
        # )
        # voxel_loss = F.binary_cross_entropy(
        #     t.sigmoid(predicted_voxels_logits),
        #     self.model.to_one_hot_voxels(predict_voxels, time=time).to(self.device),
        # )
        action_dim = predicted_actions.shape[-1]
        # predicted_action_mean_loss = t.mean(
        #     t.mean(predicted_actions[:, :, : action_dim // 2], dim=(0, 1)) ** 2
        # ) + t.mean((t.abs(predicted_actions[:, :, : action_dim // 2]) - 2) ** 2)
        # predicted_action_log_std_loss = t.mean(
        #     (t.mean(predicted_actions[:, :, action_dim // 2 :], dim=(0, 1))) ** 2
        # )
        # print(t.mean(t.abs(predicted_actions[:, :, : action_dim // 2])))

        predicted_action_mean_loss = 0
        metrics = [[], []]
        for idx, ti in enumerate(time):
            predicted_action_mean_loss += t.mean(
                t.mean(predicted_actions[idx, : ti + 1, : action_dim // 2], dim=0) ** 2
            ) + t.mean(
                (
                    t.mean(
                        t.abs(predicted_actions[idx, : ti + 1, : action_dim // 2]),
                        dim=0,
                    )
                    - 1
                )
                ** 2
            )
            metrics[0].append(
                t.mean(
                    t.mean(predicted_actions[idx, : ti + 1, : action_dim // 2], dim=0)
                    ** 2
                )
            )
            metrics[1].append(
                t.mean(
                    t.mean(
                        t.abs(predicted_actions[idx, : ti + 1, : action_dim // 2]),
                        dim=0,
                    )
                    ** 2
                )
            )
        print(predicted_actions[0, :10, :])
        print(t.mean(t.tensor(metrics[0])))
        print(t.mean(t.tensor(metrics[1])))
        predicted_action_log_std_loss = 0

        # make mean of mean=0 and mean of log_std=0
        action_regularize_loss = (
            predicted_action_mean_loss + predicted_action_log_std_loss
        )

        voxel_loss = F.mse_loss(
            t.sigmoid(predicted_voxels_logits),
            self.model.to_one_hot_voxels(predict_voxels, time=time).to(self.device),
        )

        return voxel_loss + action_regularize_loss

    def validation_step(self, batch, batch_idx):
        past_obs, predict_actions, predict_voxels, time = batch
        predict_voxels = predict_voxels[range(len(time)), time]
        # print(self.model.action_out[-1]._model[0].weight)
        (predicted_actions, predicted_voxels_logits), *_ = self.model(
            {
                "obs": past_obs[:, -1].to(
                    self.device
                ),  # dummy input, make ray model not complain
                "custom_obs": past_obs.to(self.device),
                "return_voxel": True,
            },
            None,
            None,
        )
        # Only compare the last predicted voxels
        last_predicted_voxels = t.argmax(
            predicted_voxels_logits[range(len(time)), time].reshape(
                predicted_voxels_logits.shape[0], len(self.model.materials), -1
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
    seed_everything(pretrain_config["seed"])
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
        # deterministic=True,
        precision=32,
    )
    trainer.fit(pretrainer)
    pretrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    os.makedirs(os.path.dirname(pretrain_config["weight_export_path"]), exist_ok=True)
    t.save(pretrainer.model.state_dict(), pretrain_config["weight_export_path"])
