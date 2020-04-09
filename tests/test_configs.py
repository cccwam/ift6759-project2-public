import os
from pathlib import Path

import jsonschema

from libs import helpers


def validate_configs(configs_folder):
    schema = helpers.load_dict(os.path.join(configs_folder, 'schema.json'))
    for file_name in Path(configs_folder).glob('**/*.json'):
        if file_name == 'schema.json':
            continue
        try:
            jsonschema.validate(
                schema=schema,
                instance=helpers.load_dict(file_name)
            )
        except jsonschema.exceptions.ValidationError as e:
            print(f"ValidationError for the file {config_file_path}")
            raise e


def test_validate_user_configs():
    """
    Validates that user configuration files adhere to the user configs schema
    """
    validate_configs('configs/user')
