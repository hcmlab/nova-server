import random
from ..data import ANIMALS
from ..data import COLORS

def get_key_from_request_form(request_form):
    """
    Returns the logging key from the provided request form
    Args:
        request_form (dict): Request form from Nova

    Returns:

    """
    key = request_form.get("jobID", None)

    if key is None:
        key = f"{get_random_name()}"

    return key



def get_random_name(
        combo=[COLORS, ANIMALS], separator: str = " ", style: str = "capital"
):
    if not combo:
        raise Exception("combo cannot be empty")

    random_name = []
    for word_list in combo:
        part_name = random.choice(word_list)
        if style == "capital":
            part_name = part_name.capitalize()
        if style == "lowercase":
            part_name = part_name.lower()
        if style == "uppercase":
            part_name = part_name.upper()
        random_name.append(part_name)
    return separator.join(random_name)
