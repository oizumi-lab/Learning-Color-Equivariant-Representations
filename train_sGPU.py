import json

import click
from ulid import ULID
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as tb

import constants as c
import lightning_model
from utils import train_utils
from datasets.dataloaders import parse_dataloader

@click.command()
@click.option("--exp_class")
@click.option("--exp_name")
def run(exp_class, exp_name):

    # get params
    print(c.EXPERIMENTS_PATH / (exp_class+".json"))
    with open(c.EXPERIMENTS_PATH / (exp_class+".json")) as f:
        exp_params = json.load(f)
    params_imported = exp_params[exp_name]
    params_imported["experiment_name"] = exp_class + "_" + exp_name

    # load default params
    params = train_utils.get_default_params()
    params.update(params_imported)

    # create logger
    params["ulid"] = str(ULID())
    logger = tb(
        save_dir=str(c.LOG_DIR / params["experiment_name"]),
        name="",
        version=params["ulid"],
    )

    # resume training
    resume_point = params.get("resume")
    if resume_point:
        t = lightning_model.LitHSGCNN.load_from_checkpoint(resume_point, params=params)
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
    train_loader, _, test_loader = get_data_loader(n_groups_hue, n_groups_saturation, batch_size, ours, **dataset_kwargs)

    # create lightning trainer
    if "n_epochs" not in params.keys() and "n_iters" not in params.keys():
        raise ValueError("Either n_epochs or n_iters must be specified in the experiment parameters.")
    if "n_epochs" in params.keys() and "n_iters" in params.keys():
        raise ValueError("Only one of n_epochs or n_iters can be specified in the experiment parameters.")
    
    # setup a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(logger.log_dir + "/checkpoints"),
        save_last=True,
        save_top_k=0,
    )
    
    # setup a trainer
    if "n_epochs" in params.keys():
        trainer = L.Trainer(
            #max_epochs=params["n_epochs"],
            max_epochs=3,
            devices='auto',
            logger=logger,
            callbacks=[checkpoint_callback],
        )
    else :
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(logger.log_dir + "/checkpoints"),
            filename="epoch={epoch:03d}-step={global_step}.ckpt",
            every_n_epochs=1,
            save_top_k=1,
            monitor="global_step",
            mode="max",
        )
        trainer = L.Trainer(
            max_steps=params["n_iters"],
            devices='auto',
            logger=logger,
            callbacks=[checkpoint_callback],
        )
    # train
    trainer.fit(
        t,
        train_dataloaders=train_loader,
    )
    # test
    trainer.test(
        t,
        dataloaders=test_loader,
    )

    # add to the manifest
    params["resume"] = checkpoint_callback.last_model_path
    file_path = c.MODEL_MANIFEST_DIR / f"{params['experiment_name']}_{params['ulid']}.json"
    train_utils.add_to_manifest(params, file_path)
    

if __name__ == "__main__":
    run()