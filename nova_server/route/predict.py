"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import os
from pathlib import Path, PureWindowsPath
from nova_server.utils import db_utils
from flask import Blueprint, request, jsonify
from nova_utils.ssi_utils.ssi_xml_utils import Trainer
from importlib.machinery import SourceFileLoader
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    dataset_utils,
    import_utils,
)
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from nova_utils.interfaces.server_module import Trainer as iTrainer

predict = Blueprint("predict", __name__)


@predict.route("/predict", methods=["POST"])
def predict_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = predict_data(request_form)
        status_utils.add_new_job(key, request_form=request_form)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def predict_data(request_form):
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)

    cml_dir = os.environ["NOVA_CML_DIR"]
    data_dir = os.environ["NOVA_DATA_DIR"]

    log_conform_request = dict(request_form)
    log_conform_request["password"] = "---"

    logger.info("Action 'Predict' started.")
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    trainer_file_path = Path(cml_dir).joinpath(
        PureWindowsPath(request_form["trainerFilePath"])
    )
    trainer = Trainer()

    if not trainer_file_path.is_file():
        logger.error(f"Trainer file not available: {trainer_file_path}")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    else:
        trainer.load_from_file(trainer_file_path)
        logger.info("Trainer successfully loaded.")

    if not trainer.model_script_path:
        logger.error('Trainer has no attribute "script" in model tag.')
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None

    # Load data
    try:
        update_progress(key, "Data loading")
        sessions = request_form.pop("sessions").split(";")
        roles = request_form.pop("roles").split(";")
        iterators = []
        for session in sessions:
            request_form["sessions"] = session
            if trainer.model_multi_role_input:
                request_form["roles"] = ";".join(roles)
                iterators.append(
                    dataset_utils.dataset_from_request_form(request_form, data_dir)
                )
            else:
                for role in roles:
                    request_form["roles"] = role
                    iterators.append(
                        dataset_utils.dataset_from_request_form(request_form, data_dir)
                    )

        logger.info("Data iterators initialized.")
    except ValueError as e:
        print(e)
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None

    # Load Trainer
    model_script_path = (trainer_file_path.parent / PureWindowsPath(trainer.model_script_path)).resolve()
    if (p := (model_script_path.parent / 'requirements.txt')).exists():
        with open(p) as f:
            import_utils.assert_or_install_dependencies(
                f.read().splitlines(), Path(model_script_path).stem
            )
        source = SourceFileLoader(
            "ns_cl_" + model_script_path.stem, str(model_script_path)
        ).load_module()
    # TODO: remove else block once every script is migrated to requirements.txt
    else:
        source = SourceFileLoader(
            "ns_cl_" + model_script_path.stem, str(model_script_path)
        ).load_module()
        import_utils.assert_or_install_dependencies(
            source.REQUIREMENTS, Path(model_script_path).stem
        )
    logger.info(f"Trainer module {Path(model_script_path).name} loaded")
    trainer_class = getattr(source, trainer.model_create)
    predictor = trainer_class(logger, log_conform_request)
    logger.info(f"Model {trainer.model_create} created")
    # model_script = source.TrainerClass(ds_iter, logger, request_form)

    # If the module implements the Trainer interface load weights
    if isinstance(predictor, iTrainer):
        # Load Model
        model_weight_path = (
                trainer_file_path.parent / trainer.model_weights_path
        )
        logger.info(f"Loading weights from {model_weight_path}")
        predictor.load(model_weight_path)
        logger.info("Model loaded.")

    # Iterate over all sessions
    ds_iter: HcaiNovaDynamicIterable
    for ds_iter in iterators:
        # TODO: Remove prior creation of separate iterators to reduce redundancy
        ss_ds_iter = ds_iter.to_single_session_iterator()

        logger.info("Predict data...")
        try:
            data = predictor.process_data(ss_ds_iter)
            annos = predictor.to_anno(data)
        except Exception as e:
            logger.error(str(e))
            status_utils.update_status(key, status_utils.JobStatus.ERROR)
            raise e
        logger.info("...done")

        logger.info("Saving predictions to database...")

        # TODO: Refactor to not use request form in upload
        request_form_copy = copy.copy(request_form)
        assert len(ss_ds_iter.sessions) == 1
        request_form_copy['sessions'] = ss_ds_iter.sessions[0]

        for anno in annos:
            db_utils.write_annotation_to_db(request_form_copy, anno, logger)
        logger.info("...done")

    logger.info("Prediction completed!")
    status_utils.update_status(key, status_utils.JobStatus.FINISHED)

'''Keep for later reference to implement polygons'''
    # model_script.ds_iter = ds_iter
    # model_script.request_form["sessions"] = session
    # model_script.request_form["roles"] = role
    #
    # logger.info("Execute preprocessing.")
    # model_script.preprocess()
    # logger.info("Preprocessing done.")
    #
    # logger.info("Execute prediction.")
    # model_script.predict()
    # logger.info("Prediction done.")
    #
    # logger.info("Execute postprocessing.")
    # results = model_script.postprocess()
    # logger.info("Postprocessing done.")
    #
    # logger.info("Execute saving process.")
    # db_utils.write_annotation_to_db(request_form, results, logger)
    # logger.info("Saving process done.")

    # 5. In CML case, delete temporary files..
    # if request_form["deleteFiles"] == "True":
    #     trainer_name = request_form["trainerName"]
    #     logger.info("Deleting temporary CML files...")
    #     out_dir = Path(cml_dir).joinpath(
    #         PureWindowsPath(request_form["trainerOutputDirectory"])
    #     )
    #     os.remove(out_dir / trainer.model_weights_path)
    #     os.remove(out_dir / trainer.model_script_path)
    #     for f in model_script.DEPENDENCIES:
    #         os.remove(trainer_file_path.parent / f)
    #     trainer_fullname = trainer_name + ".trainer"
    #     os.remove(out_dir / trainer_fullname)
    #     logger.info("...done")

# except Exception as e:
# logger.error('Error:' + str(e))
#   status_utils.update_status(key, status_utils.JobStatus.ERROR)
# finally:
#    del results, ds_iter, ds_iter_pp, model, model_script, model_script_path, model_weight_path, spec
