import lightning_model
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger as tb

import json
import click

import constants as c
from utils import train_utils
from datasets.dataloaders import parse_dataloader

@click.command()
@click.option("--exp_class")
@click.option("--exp_name")
def run(exp_class, exp_name):

    # get params
    with open(c.EXPERIMENTS_PATH / (exp_class+".json")) as f:
        exp_params = json.load(f)
    params_imported = exp_params[exp_name]
    params_imported["experiment_name"] = exp_class + "_" + exp_name

    # load default params
    params = train_utils.get_default_params()
    params.update(params_imported)

    # create logger
    logger = tb(save_dir=c.LOG_DIR, name=params["experiment_name"])

    resume = params["resume"]
    # resume training
    if resume != "":
        t = lightning_model.LitHSGCNN.load_from_checkpoint(resume, params=params)
    else:
        t = lightning_model.LitHSGCNN(params)

    # create data class
    data_class = params["dataset_name"]
    n_groups_hue = params["n_groups_hue"]
    n_groups_saturation = params["n_groups_saturation"]
    batch_size = params["batch_size"]
    ours = params["ours"]
    dataset_kwargs = params["dataset_kwargs"]
    get_data_loader = parse_dataloader(data_class)
    train_loader, val_loader, test_loader = get_data_loader(n_groups_hue, n_groups_saturation, batch_size, ours, **dataset_kwargs)

    # create lightning trainer
    if "n_epochs" not in params.keys() and "n_iters" not in params.keys():
        raise ValueError("Either n_epochs or n_iters must be specified in the experiment parameters.")
    if "n_epochs" in params.keys() and "n_iters" in params.keys():
        raise ValueError("Only one of n_epochs or n_iters can be specified in the experiment parameters.")
    
    if "n_epochs" in params.keys():
        trainer = L.Trainer(
            max_epochs= params["n_epochs"],
            accelerator="gpu",
            devices=4, 
            strategy="ddp",
            logger=logger)
    else :
        trainer = L.Trainer(
            max_steps= params["n_iters"],
            accelerator="gpu",
            devices=4, 
            strategy="ddp",
            logger=logger)
    trainer.fit(
        t,
        train_dataloaders = train_loader,
        val_dataloaders= val_loader,
    )
    trainer.test(
        t,
        dataloaders = test_loader,
    )

if __name__ == "__main__":
    run()