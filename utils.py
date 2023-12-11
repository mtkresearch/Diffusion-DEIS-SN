import os
import sys
import yaml
import typing
import argparse

def latest_n_checkpoints(folder, *, prefix='checkpoint', all_but=False, last_n=1):
    dirs = [d for d in os.listdir(folder) if \
        d.startswith(prefix) and not os.path.isfile(d)]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    
    if len(dirs) == 0:
        return [ ]
    
    if not all_but:
        return dirs[-last_n:]
    else:
        return dirs[:-last_n]


def _remove_defaults(args: dict, raw_args: list):
    modified_args = { }
    for k, v in args.items():
        if ('--' + k) not in raw_args:
            modified_args[k] = v
    return modified_args


def yaml_interface(script_file_path):
    def _decorator(create_parser_fn: typing.Callable):
        def _get_args_fn():
            script_file_name = os.path.basename(script_file_path)

            ret: typing.Tuple[argparse.ArgumentParser, typing.Callable] = create_parser_fn()
            parser, validation_fn = ret
            parser.add_argument(
                '--config',
                type=str,
                required=False,
                help='A yaml config file as a replacement for command line arguments'
            )
            args = parser.parse_args()

            # deleting the original args since they are not needed to go to the validation_fn
            config_file = args.config; del args.config

            combined_args = vars(args)
            if config_file:
                with open(config_file, 'r') as f:
                    yaml_dict: dict = yaml.unsafe_load(f)
                    config_from_file = yaml_dict.get('common', { })
                    config_from_file.update(yaml_dict.get(script_file_name, { }))
                
                combined_args.update(_remove_defaults(config_from_file, sys.argv[1:]))

            return validation_fn(argparse.Namespace(**combined_args))
        
        return _get_args_fn
    
    return _decorator