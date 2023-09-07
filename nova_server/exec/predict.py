"""General logic for predicting annotations to the nova database
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 06.09.2023
"""
import argparse
import json
from pathlib import Path, PureWindowsPath
from nova_utils.utils import ssi_xml_utils
from nova_utils.data.provider.nova_iterator import NovaIterator


if __name__ == '__main__':

    # parser for NOVA database connection
    nova_db_parser = argparse.ArgumentParser(description="Parse Information required to connect to the NOVA-DB", add_help=False)
    nova_db_parser.add_argument("--db_host", type=str, required=True, help="The ip-address of the NOVA-DB server")
    nova_db_parser.add_argument("--db_port", type=int, required=True, help="The ip-address of the NOVA-DB server")
    nova_db_parser.add_argument("--db_user", type=str, required=True, help="The user to authenticate at the NOVA-DB server")
    nova_db_parser.add_argument("--db_password", type=str, required=True, help="The password for the NOVA-DB server user")

    # parser for NOVA iterator
    nova_iterator_parser = argparse.ArgumentParser(description="Parse Information required to create a NovaIterator", add_help=False)
    nova_iterator_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset. Must match entries in NOVA-DB")
    nova_iterator_parser.add_argument("--data_dir", type=str, required=True, help="Path to the NOVA data directory using Windows UNC-Style")
    nova_iterator_parser.add_argument("--sessions", type=json.loads, required=True, help="Json formatted List of sessions to apply the iterator to")
    nova_iterator_parser.add_argument("--data", type=json.loads, required=True, help="Json formatted String containing dictionaries that describe the data to load")
    nova_iterator_parser.add_argument("--frame_size", type=str, help="Size of the data frame measured in time. Defaults to None")
    nova_iterator_parser.add_argument("--start", type=str, help="Start time for processing measured in time. Defaults to None")
    nova_iterator_parser.add_argument("--end", type=str, help="End time for processing measured in time. Defaults to None")
    nova_iterator_parser.add_argument("--left_context", type=str, help="Left context duration measured in time. Defaults to None")
    nova_iterator_parser.add_argument("--right_context", type=str, help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None")
    nova_iterator_parser.add_argument("--stride", type=str, help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None")
    nova_iterator_parser.add_argument("--add_rest_class", type=str, help="Whether to add a rest class for discrete annotations. Defaults to True")
    nova_iterator_parser.add_argument("--fill_missing_data", type=str, help="Whether to fill missing data. Defaults to True")

    # parser for NOVA-Server module
    nova_server_module_parser = argparse.ArgumentParser(description="Parse Information required to execute a NOVA-Server module", add_help=False)
    nova_server_module_parser.add_argument("--cml_dir", type=str, help="CML-Base directory for the NOVA-Server module")
    nova_server_module_parser.add_argument("--opt_str", type=str, help="Json formatted String containing dictionaries with key value pairs, setting the options for a NOVA-Server module")

    # main parser for predict specific options
    parser = argparse.ArgumentParser(description="Use a provided nova-server module for inference and save results to NOVA-DB", parents=[nova_db_parser, nova_iterator_parser, nova_server_module_parser])
    parser.add_argument('--trainer_file_path', type=str, required=True, help='Path to the trainer file using Windows UNC-Style')


    args, unknown = parser.parse_known_args()

    # Get all attributes that are inherited from nova_iterator_parser
    nova_iterator_attrs = {k: v for k, v in vars(args).items() if hasattr(nova_db_parser, k)}

    # Now, nova_iterator_attrs contains all the attributes from nova_iterator_parser
    print(nova_iterator_attrs)

    # Load trainer
    trainer = ssi_xml_utils.Trainer()
    trainer_file_path = Path(args.cml_dir).joinpath(PureWindowsPath(args.trainer_file_path))
    if not trainer_file_path.is_file():
        raise FileNotFoundError(f"Trainer file not available: {trainer_file_path}")
    else:
        trainer.load_from_file(trainer_file_path)
        print("Trainer successfully loaded.")

    # Load module
    if not trainer.model_script_path:
        raise ValueError('Trainer has no attribute "script" in model tag.')

    # Build data loaders
    try:
        print("Data loading")
        sessions = args.sessions
        iterators = []

        args = {}
        args |= vars(nova_db_parser.parse_known_args()[0])
        args |= vars(nova_iterator_parser.parse_known_args()[0])

        for session in sessions:
            print(session)
            ni = NovaIterator(
                **args
            )
            print(next(ni))

            # ni = NovaIterator(
            #     ip=args.ip.split(':')[0],
            #     port=args.ip.split(':')[1],
            #     user=args.user
            # )
        #     request_form["sessions"] = session
        #     if trainer.model_multi_role_input:
        #         request_form["roles"] = ";".join(roles)
        #         iterators.append(
        #             dataset_utils.dataset_from_request_form(request_form, data_dir)
        #         )
        #     else:
        #         for role in roles:
        #             request_form["roles"] = role
        #             iterators.append(
        #                 dataset_utils.dataset_from_request_form(request_form, data_dir)
        #             )
        # logger.info("Data iterators initialized.")
    except Exception as e:
        print(e)

#####
#####
#####
#####
#
# import copy
# import os
# import json
# import sys
# from pathlib import Path, PureWindowsPath
# from nova_server.utils import db_utils
# from nova_utils.utils.ssi_xml_utils import Trainer
# from importlib.machinery import SourceFileLoader
# from nova_server.utils.status_utils import update_progress
# from nova_server.utils.key_utils import get_key_from_request_form
# from nova_server.utils import (
#     status_utils,
#     log_utils,
#     dataset_utils,
#     import_utils,
# )
# from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
#     HcaiNovaDynamicIterable,
# )
# from nova_utils.interfaces.server_module import Trainer as iTrainer
#
# # Check if an argument was provided
# if len(sys.argv) != 2:
#     print("Usage: python your_script.py 'json_string'")
#     sys.exit(1)
#
# # Get the JSON string from the command line argument
# json_str = sys.argv[1]
#
# # Parse the JSON string into a dictionary
# try:
#     my_dict = json.loads(json_str)
#     print("Parsed Dictionary:")
#     print(my_dict)
# except json.JSONDecodeError as e:
#     print("Error parsing JSON:", e)
#
# key = get_key_from_request_form(request_form)
# logger = log_utils.get_logger_for_thread(key)
#
# cml_dir = os.environ["NOVA_CML_DIR"]
# data_dir = os.environ["NOVA_DATA_DIR"]
#
# log_conform_request = dict(request_form)
# log_conform_request["password"] = "---"
#
# logger.info("Action 'Predict' started.")
# status_utils.update_status(key, status_utils.JobStatus.RUNNING)
# trainer_file_path = Path(cml_dir).joinpath(
#     PureWindowsPath(request_form["trainerFilePath"])
# )
# trainer = Trainer()
#
# if not trainer_file_path.is_file():
#     logger.error(f"Trainer file not available: {trainer_file_path}")
#     status_utils.update_status(key, status_utils.JobStatus.ERROR)
#     return None
# else:
#     trainer.load_from_file(trainer_file_path)
#     logger.info("Trainer successfully loaded.")
#
# if not trainer.model_script_path:
#     logger.error('Trainer has no attribute "script" in model tag.')
#     status_utils.update_status(key, status_utils.JobStatus.ERROR)
#     return None
#
# # Load data
# try:
#     update_progress(key, "Data loading")
#     sessions = request_form.pop("sessions").split(";")
#     roles = request_form.pop("roles").split(";")
#     iterators = []
#     for session in sessions:
#         request_form["sessions"] = session
#         if trainer.model_multi_role_input:
#             request_form["roles"] = ";".join(roles)
#             iterators.append(
#                 dataset_utils.dataset_from_request_form(request_form, data_dir)
#             )
#         else:
#             for role in roles:
#                 request_form["roles"] = role
#                 iterators.append(
#                     dataset_utils.dataset_from_request_form(request_form, data_dir)
#                 )
#
#     logger.info("Data iterators initialized.")
# except ValueError as e:
#     print(e)
#     log_utils.remove_log_from_dict(key)
#     logger.error("Not able to load the data from the database!")
#     status_utils.update_status(key, status_utils.JobStatus.ERROR)
#     return None
#
# # Load Trainer
# model_script_path = (trainer_file_path.parent / PureWindowsPath(trainer.model_script_path)).resolve()
# if (p := (model_script_path.parent / 'requirements.txt')).exists():
#     with open(p) as f:
#         import_utils.assert_or_install_dependencies(
#             f.read().splitlines(), Path(model_script_path).stem
#         )
#     source = SourceFileLoader(
#         "ns_cl_" + model_script_path.stem, str(model_script_path)
#     ).load_module()
# # TODO: remove else block once every script is migrated to requirements.txt
# else:
#     source = SourceFileLoader(
#         "ns_cl_" + model_script_path.stem, str(model_script_path)
#     ).load_module()
#     import_utils.assert_or_install_dependencies(
#         source.REQUIREMENTS, Path(model_script_path).stem
#     )
# logger.info(f"Trainer module {Path(model_script_path).name} loaded")
# trainer_class = getattr(source, trainer.model_create)
# predictor = trainer_class(logger, log_conform_request)
# logger.info(f"Model {trainer.model_create} created")
# # model_script = source.TrainerClass(ds_iter, logger, request_form)
#
# # If the module implements the Trainer interface load weights
# if isinstance(predictor, iTrainer):
#     # Load Model
#     model_weight_path = (
#             trainer_file_path.parent / trainer.model_weights_path
#     )
#     logger.info(f"Loading weights from {model_weight_path}")
#     predictor.load(model_weight_path)
#     logger.info("Model loaded.")
#
# # Iterate over all sessions
# ds_iter: HcaiNovaDynamicIterable
# for ds_iter in iterators:
#     # TODO: Remove prior creation of separate iterators to reduce redundancy
#     ss_ds_iter = ds_iter.to_single_session_iterator()
#
#     logger.info("Predict data...")
#     try:
#         data = predictor.process_data(ss_ds_iter)
#         annos = predictor.to_anno(data)
#     except Exception as e:
#         logger.error(str(e))
#         status_utils.update_status(key, status_utils.JobStatus.ERROR)
#         raise e
#     logger.info("...done")
#
#     logger.info("Saving predictions to database...")
#
#     # TODO: Refactor to not use request form in upload
#     request_form_copy = copy.copy(request_form)
#     assert len(ss_ds_iter.sessions) == 1
#     request_form_copy['sessions'] = ss_ds_iter.sessions[0]
#
#     for anno in annos:
#         db_utils.write_annotation_to_db(request_form_copy, anno, logger)
#     logger.info("...done")
#
# logger.info("Prediction completed!")
# status_utils.update_status(key, status_utils.JobStatus.FINISHED)


#
# '''Keep for later reference to implement polygons'''
# # model_script.ds_iter = ds_iter
# # model_script.request_form["sessions"] = session
# # model_script.request_form["roles"] = role
# #
# # logger.info("Execute preprocessing.")
# # model_script.preprocess()
# # logger.info("Preprocessing done.")
# #
# # logger.info("Execute prediction.")
# # model_script.predict()
# # logger.info("Prediction done.")
# #
# # logger.info("Execute postprocessing.")
# # results = model_script.postprocess()
# # logger.info("Postprocessing done.")
# #
# # logger.info("Execute saving process.")
# # db_utils.write_annotation_to_db(request_form, results, logger)
# # logger.info("Saving process done.")
#
# # 5. In CML case, delete temporary files..
# # if request_form["deleteFiles"] == "True":
# #     trainer_name = request_form["trainerName"]
# #     logger.info("Deleting temporary CML files...")
# #     out_dir = Path(cml_dir).joinpath(
# #         PureWindowsPath(request_form["trainerOutputDirectory"])
# #     )
# #     os.remove(out_dir / trainer.model_weights_path)
# #     os.remove(out_dir / trainer.model_script_path)
# #     for f in model_script.DEPENDENCIES:
# #         os.remove(trainer_file_path.parent / f)
# #     trainer_fullname = trainer_name + ".trainer"
# #     os.remove(out_dir / trainer_fullname)
# #     logger.info("...done")
#
# # except Exception as e:
# # logger.error('Error:' + str(e))
# #   status_utils.update_status(key, status_utils.JobStatus.ERROR)
# # finally:
# #    del results, ds_iter, ds_iter_pp, model, model_script, model_script_path, model_weight_path, spec
