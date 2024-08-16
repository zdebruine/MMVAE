import click

from cmmvae.pipeline import(
    submit, cli, 
    merge_predictions, generate_umap,
)

@click.group()
def main():
    """Main entry point for cmmvae CLI"""

main.add_command(cli.main)
main.add_command(merge_predictions.main)
main.add_command(generate_umap.main)
main.add_command(submit.main)

if __name__ == '__main__':
    main()