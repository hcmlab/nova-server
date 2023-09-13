import os
import shutil
import imblearn
import traceback
from importlib.machinery import SourceFileLoader

from flask import Blueprint, request, jsonify
from nova_server.utils import dataset_utils, thread_utils, status_utils, log_utils, import_utils
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_job_id_from_request_form
from nova_server.utils.thread_utils import THREADS
from pathlib import Path, PureWindowsPath
from nova_utils.utils.ssi_xml_utils import Trainer
from flask import current_app

import importlib

train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_job_id_from_request_form(request_form)
        thread = train_model(request_form, current_app._get_current_object())
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_model(request_form, app_context):
    key = get_job_id_from_request_form(request_form)
    logger = log_utils.get_logger_for_job(key)
    cml_dir = app_context.config['CML_DIR']
    data_dir = app_context.config['DATA_DIR']

    try:
        status_utils.update_status(key, status_utils.JobStatus.RUNNING)
        update_progress(key, 'Initializing')

        trainer_file_path = Path(cml_dir).joinpath(PureWindowsPath(request_form["trainerFilePath"]))
        out_dir = Path(cml_dir).joinpath(
            PureWindowsPath(request_form["trainerOutputDirectory"])
        )
        trainer_name = request_form["trainerName"]

        log_conform_request = dict(request_form)
        log_conform_request['password'] = '---'

        logger.info("Action 'Train' started.")
        trainer = Trainer()

        if not trainer_file_path.is_file():
            logger.error(f"Trainer file not available: {trainer_file_path}")
            status_utils.update_status(key, status_utils.JobStatus.ERROR)
            return None
        else:
            trainer.load_from_file(trainer_file_path)
            logger.info("Trainer successfully loaded.")


        # Load Data
        try:
            update_progress(key, 'Data loading')
            ds_iter = dataset_utils.dataset_from_request_form(request_form, data_dir)
            logger.info("Train-Data successfully loaded...")
        except ValueError as e:
            print(e)
            log_utils.remove_log_from_dict(key)
            logger.error("Not able to load the data from the database!")
            status_utils.update_status(key, status_utils.JobStatus.ERROR)
            return None

        # Load Trainer
        model_script_path = trainer_file_path.parent / trainer.model_script_path
        source = SourceFileLoader("model_script", str(model_script_path)).load_module()
        import_utils.assert_or_install_dependencies(source.REQUIREMENTS, Path(model_script_path).stem)
        model_script = source.TrainerClass(ds_iter, logger, request_form)

        # Set Options 
        logger.info("Setting options...")
        if not request_form["OptStr"] == '':
            for k, v in dict(option.split("=") for option in request_form["OptStr"].split(";")).items():
                if v in ('True', 'False'):
                    model_script.OPTIONS[k] = True if v == 'True' else False
                elif v == 'None':
                    model_script.OPTIONS[k] = None
                else:
                    model_script.OPTIONS[k] = v
                logger.info(k + '=' + v)
        logger.info("...done.")

        logger.info("Preprocessing data...")
        # TODO: generic preprocessing interface, remove request form from model interface
        model_script.preprocess()
        logger.info("...done")

        logger.info("Training model...")
        model_script.train()
        logger.info("...done")

        # Save
        logger.info('Saving...')
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.info_trained = True

        # TODO add users / sessions
        # TODO there might be a more elegant way to get classes...
        tmp = ds_iter.annos[ds_iter.roles[0] + "." + ds_iter.schemes[0]].labels
        trainer.classes = []
        if request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
            for entry in tmp:
                trainer.classes.append({"name": tmp[entry][0]})
        else:
            for entry in tmp:
                trainer.classes.append({"name": tmp[entry]})

        weight_path = model_script.save(out_dir / trainer_name)
        trainer.model_weights_path = os.path.basename(weight_path)
        logger.info('...weights')
        shutil.copy(model_script_path, out_dir / trainer.model_script_path)
        logger.info('...train script')
        for f in model_script.DEPENDENCIES:
            shutil.copy(trainer_file_path.parent / f, out_dir / f)
        logger.info('...dependencies')
        trainer.write_trainer_to_file(out_dir / trainer_name)
        logger.info('...trainerfile')
        logger.info("Training completed!")
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)

    except Exception as e:
        logger.error('Error:' + traceback.format_exc())
        status_utils.update_status(key, status_utils.JobStatus.ERROR)


# TODO DATA BALANCING
def balance_data(request_form, x_np, y_np):
    # DATA BALANCING
    if request_form["balance"] == "over":
        print("OVERSAMPLING from {} Samples".format(x_np.shape))
        oversample = imblearn.over_sampling.SMOTE()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))

    if request_form["balance"] == "under":
        print("UNDERSAMPLING from {} Samples".format(x_np.shape))
        undersample = imblearn.under_sampling.RandomUnderSampler()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))
