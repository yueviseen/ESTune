import json


def read_performance_from_json(filename):
    """
    Reads a JSON file and returns the value of the 'perfomance' field.

    :param filename: Path to the JSON file.
    :return: The value of the 'perfomance' field, or None if not found.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('perfomance')
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
