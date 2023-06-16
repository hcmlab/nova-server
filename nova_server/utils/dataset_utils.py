from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)


def dataset_from_request_form(request_form, data_dir):
    """
    Creates a tensorflow dataset from nova dynamically
    :param request_form: the requestform that specifices the parameters of the dataset
    """
    db_config_dict = {
        "ip": request_form["dbServer"].split(":")[0],
        "port": int(request_form["dbServer"].split(":")[1]),
        "user": request_form["dbUser"],
        "password": request_form["dbPassword"],
    }

    flattenSamples = False
    if request_form.get("flattenSamples") == "true":
        flattenSamples = True

    ds_iter = HcaiNovaDynamicIterable(
        # Database Config
        db_config_path=None,  # os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.cfg'),
        db_config_dict=db_config_dict,
        # Dataset Config
        dataset=request_form.get("database"),
        nova_data_dir=data_dir,
        sessions=request_form.get("sessions").split(";")
        if request_form.get("sessions")
        else None,
        roles=request_form.get("roles", "").split(";")
        if request_form.get("roles")
        else None,
        schemes=request_form.get("scheme", "").split(";")
        if request_form.get("scheme")
        else None,
        annotator=request_form.get("annotator"),
        data_streams=request_form.get("streamName", "").split(";")
        if request_form.get("streamName")
        else None,
        # Sample Config
        frame_size=request_form.get("frameSize"),
        stride=request_form.get("stride"),
        left_context=request_form.get("leftContext"),
        right_context=request_form.get("rightContext"),
        start=request_form.get("startTime"),
        end=request_form.get("endTime"),
        # TODO: This does not work with pytorch bridge when set to true because the data field does not contain the role anymore<.
        # transformation cannot be applied. fix it!
        flatten_samples=flattenSamples,
        supervised_keys=[
            request_form.get("streamName", "").split(";")[0],
            request_form.get("scheme", "").split(";")[0],
        ],
        # Additional Config
        clear_cache=True,
    )

    return ds_iter
