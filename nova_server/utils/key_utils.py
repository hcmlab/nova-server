

def get_key_from_request_form(request_form):

    id_components = [
        request_form.get('username', None),
        request_form.get('database', None),
        request_form.get('scheme', None),
        request_form.get('streamName', None),
        request_form.get('annotator', None),
        request_form.get('sessions', None).replace(";", "_"),
    ]
    server_key = '_'.join([x for x in id_components if x])


    # serverkey = request_form['username'] + '_' + request_form['database'] + '_' + request_form['scheme'] + '_' + \
    #     request_form['streamName'] + '_' + request_form['annotator'] + '_' + \
    #     request_form['sessions'].replace(";", "_")
    return server_key[:128]
