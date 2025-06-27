import os
import argparse
import yaml
import shutil


def get_model_name(args):
    model_name = [args.exp_name, 
                  'v' + str(args.version)]
    return '__'.join(model_name)


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge 'override' dict into 'base' dict.
    The 'override' values take precedence if there's a conflict.
    """
    for key, value in override.items():
        if (
            key in base 
            and isinstance(base[key], dict) 
            and isinstance(value, dict)
        ):
            # Both base[key] and override[key] are dicts -> recurse
            deep_merge(base[key], value)
        else:
            # Otherwise, just override
            base[key] = value
    return base

def load_yaml_with_base(file_path: str) -> dict:
    """
    Load a YAML file that may optionally specify a '_BASE_' file to inherit from.
    If '_BASE_' is present, load that base file first and then merge this file's
    content on top.
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)  # config is a dict or None

    if not isinstance(config, dict):
        # Ensure we have a top-level dictionary; otherwise raise an error or handle differently
        raise ValueError(f"The file {file_path} doesn't contain a valid top-level dictionary.")

    # Check if a _BASE_ key exists; pop it so it won't remain in the final config
    base_path = config.pop('_BASE_', None)

    if base_path:
        # Handle relative path: interpret relative to the current YAML file's directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(file_path), base_path)

        # Recursively load the base file
        base_config = load_yaml_with_base(base_path)

        # Merge the current file's config into the base config
        final_config = deep_merge(base_config, config)
    else:
        # No _BASE_ specified, so this config is final
        final_config = config

    return final_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help="Path to the YAML configuration file")

    args, remaining_args = parser.parse_known_args()
    config = load_yaml_with_base(args.config)

    for key, value in config.items():
        if isinstance(value, list):  
            parser.add_argument(f'--{key}', nargs='+', type=type(value[0]), default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args(remaining_args)
    args.model_name = get_model_name(args)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.checkpoint_dir = os.path.join(args.output_dir, 'models')
    args.save_config_file = os.path.join(args.output_dir, 'config.yaml')    
    return args

