import random

def get_key_from_request_form(request_form):
    """
    Returns the logging key from the provided request form
    Args:
        request_form (dict): Request form from Nova

    Returns:

    """
    key = request_form.get("jobID", None)

    if key is None:
        key = f"local_{random.randint(0, 10 ^ 7)}"

    # id_components = [
    #     request_form.get('username', None),
    #     request_form.get('database', None),
    #     request_form.get('scheme', None),
    #     request_form.get('streamName', None),
    #     request_form.get('annotator', None),
    #     request_form.get('sessions', None).replace(";", "_"),
    # ]
    # server_key = '_'.join([x for x in id_components if x])


    # serverkey = request_form['username'] + '_' + request_form['database'] + '_' + request_form['scheme'] + '_' + \
    #     request_form['streamName'] + '_' + request_form['annotator'] + '_' + \
    #     request_form['sessions'].replace(";", "_")
    return key
