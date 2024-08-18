import os
import click
import inspect
import warnings

def parse_env_var_name(f, *param_decls, **attrs):
    module_name = inspect.getmodule(f).__name__
    option_name = param_decls[-1].lstrip('--') if param_decls else 'unknown_option'
    """Generate environment variable name based on the option name."""
    env_var = f"{module_name.upper()}_{option_name.lstrip('-').replace('-', '_').upper()}".replace('.', '_')

    return env_var, module_name, option_name

def click_env_option(*param_decls, **attrs):
    """Custom click option decorator that retrieves default values from environment variables."""
    def decorator(f):
        # Get the option name from the parameter declarations
        if not 'envvar' in attrs:
            env_name, *f_info = parse_env_var_name(f, *param_decls, **attrs)
            attrs['envvar'] = env_name
        # Apply the original click option decorator
        click_option = click.option(*param_decls, **attrs)
        return click_option(f)

    return decorator
