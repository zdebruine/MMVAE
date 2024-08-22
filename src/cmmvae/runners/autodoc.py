"""
    CMMVAE Command to automatically build and serve documentation using pdoc.
"""

from typing import Literal, Union
import click

import os
import subprocess
import threading
import shutil
from http.server import SimpleHTTPRequestHandler, HTTPServer


def replace_placeholders(root_dir, repo_owner, repo_name):
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()

            # Replace placeholders
            content = content.replace("&lt;GIT_REPO_OWNER&gt;", repo_owner)
            content = content.replace("&lt;GIT_REPO_NAME&gt;", repo_name)

            # Write the modified content back to the file
            with open(file_path, "w") as f:
                f.write(content)


class _QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler to silence logging for HTTP requests."""

    def log_message(self, format, *args):
        pass


def _clean_directory(directory_path):
    # Remove the entire directory
    shutil.rmtree(directory_path)
    # Recreate the directory
    os.makedirs(directory_path)


def _rebuild_pdoc(source_dir, output_dir):
    """Rebuild the pdoc documentation."""
    try:
        subprocess.run(
            [
                "pdoc",
                source_dir,
                "--output",
                output_dir,
                "--docformat",
                "google",
                "--show-source",
                "--no-include-undocumented",
                "-e",
                "cmmvae=https://GIT.com/zdebruine/MMVAE/",
            ],
            check=True,
        )

        repo_owner = os.getenv("GIT_REPO_OWNER")
        repo_name = os.getenv("GIT_REPO_NAME")

        if repo_owner and repo_name:
            replace_placeholders("./.cmmvae/docs", repo_owner, repo_name)
            print(f"Successfully rebuilt pdoc documentation in {output_dir}")
        else:
            raise RuntimeError(
                "Environment variables GIT_REPO_OWNER or GIT_REPO_NAME are not set."
            )

    except subprocess.CalledProcessError as e:
        print(f"Error during pdoc rebuild: {e}")


def _start_server(
    source_dir: str, output_dir: str, port: Union[Literal["auto"], int] = 8000
):
    """Start an HTTP server serving files from the specified directory."""

    try:
        handler = _QuietHTTPRequestHandler
        httpd = HTTPServer(
            server_address=("", port),
            RequestHandlerClass=lambda *args, **kwargs: handler(
                *args, directory=output_dir, **kwargs
            ),
        )
        print(f"Serving HTTP on port {port} from directory {output_dir}...")
    except OSError as e:
        if e.errno == 98:
            print(f"HTTP port {port} in use.")
            exit(1)
        raise e
    # Run the server in a separate thread to
    # allow for rebuilds in the main thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        while True:
            input_msg = (
                "Commands:\n\t"
                "(R|build): rebuild pdoc documentation\n\t"
                "(C|clean): remove all files in .cmmvae/docs"
            )
            command = input(input_msg).lower()
            if command in ("r", "build", "rebuild"):
                print("Rebuilding pdoc documentation...")
                _rebuild_pdoc(source_dir, output_dir)
            elif command in ("c", "clean"):
                print(f"Cleaning {output_dir}")
                _clean_directory(output_dir)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()
        server_thread.join()


@click.group()
def autodoc():
    """Build or serve documention automatically using pdoc."""


@autodoc.command()
@click.option("--port", default=8000, help="Port to open http.server")
@click.option(
    "--source_dir", default="./cmmvae", show_default=True, help="Directory of module"
)
@click.option(
    "--output_dir",
    default="./.cmmvae/docs",
    show_default=True,
    help="Directory to store documentation",
)
def serve(source_dir, output_dir, port):
    """Serve documention with http server."""
    _start_server(source_dir, output_dir, port=port)


@autodoc.command()
@click.option(
    "--source_dir", default="./cmmvae", show_default=True, help="Directory of module"
)
@click.option(
    "--output_dir",
    default="./.cmmvae/docs",
    show_default=True,
    help="Directory to store documentation",
)
def build(source_dir, output_dir):
    """Build module documentation to directory with pdoc."""
    _rebuild_pdoc(source_dir, output_dir)


if __name__ == "__main__":
    autodoc()
