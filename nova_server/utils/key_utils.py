

def get_key_from_request_form(request_form):
    return request_form.get('database') + '_' + request_form.get('scheme') + '_' + \
           request_form.get('stream') + '_' + request_form.get('annotator') + '_' + \
           request_form.get('sessions') + '_' + request_form.get('trainerScriptName') + '_' + \
           request_form.get('mode')
