import click
import inspect

import cmmvae.runners

@click.group()
def main():
    """Main entry point for cmmvae CLI"""

for name, obj in inspect.getmembers(cmmvae.runners):
    if isinstance(obj, click.Command):
        main.add_command(obj)

if __name__ == '__main__':
    main()