import os
import re

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config file read by the pipeline. Set the environment variable, use
# `python code/mm_gw.py --config file.yaml`, or overwrite this variable before
# importing mm_gw_pipeline.
CONFIG_FILE = os.environ.get("BRIGHT_COSMOLOGY_CONFIG",
                             os.path.join(REPO_ROOT, "code", "inputs.yaml"))


class SciYearLoader(yaml.SafeLoader):
    pass


SciYearLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?{year_re}
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+){year_re}
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?{year_re}
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*{year_re}
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$'''.format(year_re="([yY])?"), re.X),
    list(u'-+0123456789.'))


def load_yaml(path_to_input=None, raw=None):
    """
    Imports the input parameters from a file and returns a dictionary (
    the values of the different keys are the ones required by the main)

    Parameters
    ----------
    path_to_input : string of chars

    Returns
    ----------
    my_dictionary : dictionary

    """
    if not raw:  # if raw is not None
        with open(path_to_input or CONFIG_FILE, 'r') as stream:
            raw = "".join(stream.readlines())

    return yaml.load(raw, Loader=SciYearLoader)

def input_dict(path_to_input=None):
    return load_yaml(path_to_input=path_to_input)


def read():
    new_dictionary = input_dict()
    input_dictionary = {}

    print("Here are the input dictionaries: \n")
    for key_1 in new_dictionary.keys():
        print(key_1)
        input_dictionary = dict(input_dictionary, **new_dictionary[key_1])

        for key, value in new_dictionary[key_1].items():
            print("{} : {}".format(key, value))
        print()

    return input_dictionary

def write_in_file(path):

    with open(path, 'w') as file:

        new_dictionary = input_dict()

        for key_1 in new_dictionary.keys():
                    
            file.writelines(f"{key_1}\n")
            

            for key, value in new_dictionary[key_1].items():
                file.writelines('\t')
                file.writelines(f"{key} : {value}\n")
        file.writelines(' ')