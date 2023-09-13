"""General logic for predicting annotations to the nova database
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 06.09.2023
"""
import argparse
import json
from pathlib import Path, PureWindowsPath
from nova_utils.utils import ssi_xml_utils
from nova_utils.data.provider.nova_iterator import NovaIterator
from nova_utils.data.handler.mongo_handler import AnnotationHandler
from importlib.machinery import SourceFileLoader


# Parser for NOVA database connection
nova_db_parser = argparse.ArgumentParser(
    description="Parse Information required to connect to the NOVA-DB", add_help=False
)
nova_db_parser.add_argument(
    "--db_host", type=str, required=True, help="The ip-address of the NOVA-DB server"
)
nova_db_parser.add_argument(
    "--db_port", type=int, required=True, help="The ip-address of the NOVA-DB server"
)
nova_db_parser.add_argument(
    "--db_user",
    type=str,
    required=True,
    help="The user to authenticate at the NOVA-DB server",
)
nova_db_parser.add_argument(
    "--db_password",
    type=str,
    required=True,
    help="The password for the NOVA-DB server user",
)

# Parser for NOVA iterator
nova_iterator_parser = argparse.ArgumentParser(
    description="Parse Information required to create a NovaIterator", add_help=False
)
nova_iterator_parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset. Must match entries in NOVA-DB",
)
nova_iterator_parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the NOVA data directory using Windows UNC-Style",
)
nova_iterator_parser.add_argument(
    "--sessions",
    type=json.loads,
    required=True,
    help="Json formatted List of sessions to apply the iterator to",
)
nova_iterator_parser.add_argument(
    "--data",
    type=json.loads,
    required=True,
    help="Json formatted String containing dictionaries that describe the data to load",
)
nova_iterator_parser.add_argument(
    "--frame_size",
    type=str,
    help="Size of the data frame measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--start",
    type=str,
    help="Start time for processing measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--end", type=str, help="End time for processing measured in time. Defaults to None"
)
nova_iterator_parser.add_argument(
    "--left_context",
    type=str,
    help="Left context duration measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--right_context",
    type=str,
    help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--stride",
    type=str,
    help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--add_rest_class",
    type=str,
    help="Whether to add a rest class for discrete annotations. Defaults to True",
)
nova_iterator_parser.add_argument(
    "--fill_missing_data",
    type=str,
    help="Whether to fill missing data. Defaults to True",
)

# Parser for NOVA-Server module
nova_server_module_parser = argparse.ArgumentParser(
    description="Parse Information required to execute a NOVA-Server module",
    add_help=False,
)
nova_server_module_parser.add_argument(
    "--cml_dir", type=str, help="CML-Base directory for the NOVA-Server module"
)
nova_server_module_parser.add_argument(
    "--opt_str",
    type=str,
    help="Json formatted String containing dictionaries with key value pairs, setting the options for a NOVA-Server module",
)

# Main parser for predict specific options
parser = argparse.ArgumentParser(
    description="Use a provided nova-server module for inference and save results to NOVA-DB",
    parents=[nova_db_parser, nova_iterator_parser, nova_server_module_parser],
)
parser.add_argument(
    "--trainer_file_path",
    type=str,
    required=True,
    help="Path to the trainer file using Windows UNC-Style",
)

if __name__ == "__main__":

    # Parse arguments
    args, _ = parser.parse_known_args()
    # print(args)

    # Create argument groups
    db_args, _ = nova_db_parser.parse_known_args()
    iter_args, _ = nova_iterator_parser.parse_known_args()
    module_args, _ = nova_server_module_parser.parse_known_args()

    caught_ex = False

    # Load trainer
    trainer = ssi_xml_utils.Trainer()
    trainer_file_path = Path(module_args.cml_dir).joinpath(
        PureWindowsPath(args.trainer_file_path)
    )
    if not trainer_file_path.is_file():
        raise FileNotFoundError(f"Trainer file not available: {trainer_file_path}")
    else:
        trainer.load_from_file(trainer_file_path)
        print("Trainer successfully loaded.")

    # Load module
    if not trainer.model_script_path:
        raise ValueError('Trainer has no attribute "script" in model tag.')

    # Build data loaders
    sessions = iter_args.sessions
    iterators = []

    # TODO split for role if multirole input is false

    args = {**vars(db_args), **vars(iter_args)}

    for session in sessions:
        print(session)
        args["sessions"] = [session]
        ni = NovaIterator(**args)
        iterators.append(ni)
    print("Data iterators initialized")

    # Load Trainer
    model_script_path = (
        trainer_file_path.parent / PureWindowsPath(trainer.model_script_path)
    ).resolve()
    source = SourceFileLoader(
        "ns_cl_" + model_script_path.stem, str(model_script_path)
    ).load_module()
    print(f"Trainer module {Path(model_script_path).name} loaded")

    trainer_class = getattr(source, trainer.model_create)
    predictor = trainer_class()
    print(f"Model {trainer.model_create} created")

    # TODO implement SSI interface
    # If the module implements the Trainer interface load weights
    # if isinstance(predictor, iTrainer):
    #     # Load Model
    #     model_weight_path = (
    #             trainer_file_path.parent / trainer.model_weights_path
    #     )
    #     logger.info(f"Loading weights from {model_weight_path}")
    #     predictor.load(model_weight_path)
    #     logger.info("Model loaded.")

    # Init database handler
    db_handler = AnnotationHandler(**vars(db_args))

    # Iterate over all sessions
    for ds_iter in iterators:

        print(f"Predict session {ds_iter.sessions[0]}...")
        try:
            data = predictor.process_data(ds_iter)
            annos = predictor.to_anno(data)
        except FileNotFoundError as e:
            print(
                f"\tIterator exited with error: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

        print("Saving predictions to database...")

        try:
            for anno in annos:
                db_handler.save(anno)
        except FileExistsError as e:
            print(
                f"\tCould note save annotation: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

    print("Prediction completed!")
    if caught_ex:
        print(
            "Prediction job encountered errors for some sessions. Check logs for details."
        )
        exit(1)
