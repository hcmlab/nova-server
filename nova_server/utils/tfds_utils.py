import hcai_datasets
import tensorflow_datasets as tfds

def dataset_from_request(request):
    """
    Creates a tensorflow dataset from nova dynamically
    :param request_form: the requestform that specifices the parameters of the dataset
    """
    db_config_dict = {
        'ip': request.form.get("server").split(':')[0],
        'port': int(request.form.get("server").split(':')[1]),
        'user': request.form.get("username"),
        'password': request.form.get("password")
    }

    ds, ds_info = tfds.load(
        'hcai_nova_dynamic',
        split='dynamic_split',
        with_info=True,
        as_supervised=False,
        data_dir='.',
        read_config=tfds.ReadConfig(
            shuffle_seed=1337
        ),
        builder_kwargs={
            # Database Config
            'db_config_path': None,  # os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.cfg'),
            'db_config_dict': db_config_dict,

            # Dataset Config
            'dataset': request.form.get("database"),
            'nova_data_dir': request.form.get("dataPath"),
            'sessions': request.form.get("sessions").split(';'),
            'roles': request.form.get("roles").split(';'),
            'schemes': request.form.get("scheme").split(';'),
            'annotator': request.form.get("annotator"),
            'data_streams': request.form.get("stream").split(' '),

            # Sample Config
            'frame_size': 0.04,
            'left_context': 0,
            'right_context': 0,
            'start': 0,
            'end': 0,
            'flatten_samples': True,
            'supervised_keys': [request.form.get("stream").split(' ')[0],
                                request.form.get("scheme").split(';')[0]],

            # Additional Config
            'clear_cache': True,
        }
    )

    return ds, ds_info
