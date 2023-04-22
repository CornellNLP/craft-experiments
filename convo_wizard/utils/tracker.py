import json
import logging
import os

import jsonlines
import wandb
from rich.console import Console
from rich.logging import RichHandler


class Tracker(object):
    def __init__(self, config, base_path_to_store_results, experiment_name, project_name='convo_wizard',
                 entity_name='cornell-nlp', log_to_wandb=True, log_level=logging.DEBUG):
        super().__init__()

        self._base_path_to_store_results = base_path_to_store_results
        self._config = config
        self._experiment_name = experiment_name
        self._project_name = project_name
        self._entity_name = entity_name
        self._log_to_wandb = log_to_wandb
        self._log_level = log_level

        self._setup()

    def _setup(self):
        # Create base folders to store results.
        self._run_path = os.path.join(self._base_path_to_store_results, self._project_name, self._experiment_name)
        self._checkpoints_path = os.path.join(self._run_path, 'checkpoints')
        os.makedirs(self._run_path, exist_ok=True)
        os.makedirs(self._checkpoints_path, exist_ok=True)

        # Store the config in the base folder.
        config_path = os.path.join(self._run_path, 'config.json')
        with open(config_path, 'w') as fp:
            json.dump(self._config, fp)

        # Initialize the logger.
        log_path = os.path.join(self._run_path, 'log.txt')
        logging.basicConfig(level=self._log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(log_path), RichHandler(console=Console(quiet=False))])

        # Initialize wandb logger.
        if self._log_to_wandb:
            self._wandb_run = wandb.init(entity=self._entity_name, project=self._project_name,
                                         name=self._experiment_name, config=self._config)

    def log_preds(self, epoch, split_name, preds):
        pass

    def log_metrics(self, epoch, split_name, metrics):
        splitwise_metrics_file = os.path.join(self._run_path, f'{split_name}_split_metrics.jsonl')
        metrics_ = {'epoch': epoch, 'metrics': metrics}
        with jsonlines.open(splitwise_metrics_file, 'a') as fp:
            fp.write(metrics_)

        # Write to wandb.
        if self._log_to_wandb:
            metrics_ = {f'{split_name}/{metric_key}': value for metric_key, value in metrics.items()}
            metrics_['epoch'] = epoch
            wandb.log(metrics_)

        # Log to console.
        logging.info(f'{split_name} metrics: {metrics_}')

    def save_model(self, model):
        model_path = os.path.join(self._run_path, 'model.pt')
        model.save_pretrained(model_path)

    def save_checkpoint(self, trainer, epoch):
        checkpoint_path = os.path.join(self._checkpoints_path, f'checkpoint_{epoch}.pt')
        trainer.save_checkpoint(epoch, checkpoint_path)

    def done(self):
        if self._log_to_wandb:
            wandb.finish()
