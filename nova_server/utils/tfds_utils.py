import sys

sys.path.append('/Users/Marco/Documents/Uni/Masterarbeit/hcai_datasets')
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable
from nova_server.utils.path_config import data_dir

def dataset_from_request_form(request_form, mode="train"):
    """
    Creates a tensorflow dataset from nova dynamically
    :param mode: train or predict
    :param request_form: the requestform that specifices the parameters of the dataset
    """
    db_config_dict = {
        'ip': request_form["server"].split(':')[0],
        'port': int(request_form["server"].split(':')[1]),
        'user': request_form["username"],
        'password': request_form["password"]
    }

    if mode == "train":
        start = request_form["startTime"]
        end = request_form["cmlBeginTime"]
    else:
        start = request_form["cmlBeginTime"]
        end = request_form["cmlEndTime"]

    # ToDo WTF?
    if end == '-1':
        end = None

    ds_iter = HcaiNovaDynamicIterable(
        # Database Config
        db_config_path=None,  # os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.cfg'),
        db_config_dict=db_config_dict,

        # Dataset Config
        dataset=request_form["database"],
        #nova_data_dir=request_form["dataPath"],
        nova_data_dir=data_dir,
        sessions=request_form["sessions"].split(';'),
        roles=request_form["roles"].split(';'),
        schemes=request_form["scheme"].split(';'),
        annotator=request_form["annotator"],
        data_streams=request_form["streamName"].split(' '),

        # Sample Config
        frame_size=request_form["sampleRate"],
        left_context=request_form["leftContext"],
        right_context=request_form["rightContext"],
        start=start,
        end=end,
        flatten_samples=True,
        supervised_keys=[request_form["streamName"].split(' ')[0],
                         request_form["scheme"].split(';')[0]],

        # Additional Config
        clear_cache=True,
    )

    return ds_iter
