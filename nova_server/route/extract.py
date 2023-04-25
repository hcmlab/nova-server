"""This file contains the general logic for extracting feature streams to the filesystem"""

from flask import Blueprint, request, jsonify
from nova_server.utils import (
    dataset_utils,
    thread_utils,
    status_utils,
    log_utils,
    import_utils,
)
from nova_server.utils.key_utils import get_key_from_request_form
from flask import current_app
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from pathlib import Path, PureWindowsPath
from nova_utils.db_utils.nova_types import DataTypes, AnnoTypes
from nova_utils.ssi_utils.ssi_xml_utils import Chain, ChainLink
from nova_utils.ssi_utils.ssi_stream_utils import Stream, Chunk, FileTypes, NPDataTypes
from importlib.machinery import SourceFileLoader
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from soundfile import write


extract = Blueprint("extract", __name__)


@extract.route("/extract", methods=["POST"])
def extract_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = extract_data(request_form, current_app._get_current_object())
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def extract_data(request_form, app_context):

    # Initialize
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)
    cml_dir = app_context.config["CML_DIR"]
    data_dir = app_context.config["DATA_DIR"]

    chain_file_path = Path(cml_dir).joinpath(
        PureWindowsPath(request_form["chainFilePath"])
    )

    log_conform_request = dict(request_form)
    log_conform_request["password"] = "---"

    logger.info("Action 'Extract' started.")
    chain = Chain()

    # Load chain
    # TODO: update chain file path from request form
    if not chain_file_path.is_file():
        logger.error("Chain file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    else:
        chain.load_from_file(chain_file_path)
        logger.info("Chain successfully loaded.")

    # At the moment we support only one link in the extractor
    if len(chain.links) > 1:
        raise AssertionError(
            "Loaded chainfile consists of more than one processing step. Currently only one processing step is supported."
        )

    # Check if chain expects inputs for all role in one sample
    multi_role_input = chain.links[0].multi_role_input

    # Load data
    logger.info("Initializing data iterators...")
    try:
        update_progress(key, "Data loading")
        sessions = request_form.pop("sessions").split(";")
        roles = request_form.pop("roles").split(";")
        iterators = []
        for session in sessions:
            request_form["sessions"] = session
            if multi_role_input:
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

        logger.info("...done")
    except ValueError as e:
        print(e)
        log_utils.remove_log_from_dict(key)
        logger.error(e)
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    except FileNotFoundError as e:
        print(e)
        log_utils.remove_log_from_dict(key)
        logger.error(e)
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None

    # Iterate over all sessions
    ds_iter: HcaiNovaDynamicIterable
    for ds_iter in iterators:

        # Iterate over all chain links
        cl: ChainLink
        for i, cl in enumerate(chain.links):

            # Load chain link module
            model_script_path = chain_file_path.parent / cl.script
            source = SourceFileLoader(
                "ns_cl_" + model_script_path.stem, str(model_script_path)
            ).load_module()
            import_utils.assert_or_install_dependencies(
                source.REQUIREMENTS, Path(model_script_path).stem
            )
            logger.info(f"Extraction module {Path(model_script_path).name} loaded")
            extractor_class = getattr(source, cl.create)
            extractor = extractor_class(logger, log_conform_request)
            logger.info(f"Extractor {cl.create} created")

            # Set Options
            logger.info("Setting options...")
            if request_form.get("optStr"):
                for k, v in [
                    option.split("=") for option in request_form["optStr"].split(";")
                ]:
                    if v in ("True", "False"):
                        extractor.options[k] = True if v == "True" else False
                    elif v == "None":
                        extractor.options[k] = None
                    else:
                        extractor.options[k] = v
                    logger.info(k + "=" + v)
            logger.info("...done.")

            # Check if there will be more chain links to execute
            if i + 1 != len(chain.links):

                # Assert chainability
                if not extractor.chainable:
                    raise AssertionError(
                        "Extraction module does not not support further chaining but is not the last link in chain."
                    )

                # TODO implement nova data types in the server module
                # TODO process data and write tmp session files to disk if necessary
                logger.info("Extract data...")
                data = extractor.process_data(ds_iter)
                ds_iter = extractor.to_ds_iterable(data)
                logger.info("...done")

            # Last element of chain
            else:
                logger.info("Extract data...")
                data = extractor.process_data(ds_iter)
                stream_dict = extractor.to_stream(data)
                logger.info("...done")

                # Write to disk
                if not stream_dict:
                    raise ValueError("Stream data is none")
                else:
                    logger.info("Write data to disk...")
                    for stream_id, (data_type, sr, data) in stream_dict.items():

                        suffix = request_form.get("suffix", default='')
                        out_path = Path(ds_iter.nova_data_dir) / ds_iter.dataset / ds_iter.sessions[0] / stream_id + '_' + suffix

                        logger.info(f"- {out_path}")

                        # SSI-Stream
                        if data_type == DataTypes.FEATURE:
                            ftype = FileTypes.BINARY
                            sr = int(sr)
                            dim = data.shape[-1] if len(data.shape) > 1 else 1
                            dtype = NPDataTypes(type(data.dtype).type)
                            chunks = [
                                Chunk(f=0, t=data.shape[0] / sr, b=0, n=data.shape[0])
                            ]
                            ssi_stream = Stream(
                                ftype=ftype,
                                sr=sr,
                                dim=dim,
                                byte=data.dtype.itemsize,
                                dtype=dtype,
                                data=data,
                                chunks=chunks,
                            )

                            ssi_stream.save(out_path)

                        # Audio Data
                        elif data_type == DataTypes.AUDIO:
                            out_path = out_path.parent / (out_path.name + '.wav')
                            write(file=out_path, data=data, samplerate=sr)
                        # Video Data
                        elif data_type == DataTypes.VIDEO:
                            raise NotImplementedError()

                        # Annotations
                        elif data_type in AnnoTypes:
                            raise NotImplementedError()
                    logger.info("...done")

        logger.info("Extraction completed!")
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)

        logger.info("Action 'Extract' finished.")
        # TODO: Start legacy ssi chain support
