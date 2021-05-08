'''
command line interface
'''
from typing import Optional
from typing import Tuple
import click

from qanta import util
from qanta.http_service import create_app
from qanta.model_proxy import ModelProxy


@click.group()
def cli():
    pass


@cli.command()
@click.option('--config-file', default='conf/BM25-Retriever.yaml')
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(
    config_file: str,
    host: Optional[str] = '0.0.0.0',
    port: Optional[int] = 4816,
    disable_batch: Optional[bool] = False,
):
    """
    Start web server wrapping the model
    Args:
        config_file: str, path to the yaml file for model configuration
        host: str, server host name
        port: int, server port number
        disable_batch: bool, if batch evaluation is enabled
    """
    app = create_app(config_file=config_file, enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
@click.option('--config-file', default='conf/BM25-Retriever.yaml')
def train_retriever(config_file: str):
    """
    Train a retriever
    Args:
        config_file: str, path to the yaml file for model configuration
    """
    ModelProxy.train_retriever(config_file)


@cli.command()
@click.option('--config-file', default='conf/TFIDF-None.yaml')
def train_reranker(config_file: str):
    """
    Train a reranker
    Args:
        config_file: str, path to the yaml file for model configuration
    """
    ModelProxy.train_reranker(config_file)


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(
    local_qanta_prefix: Optional[str] = 'data/',
    retrieve_paragraphs: Optional[bool] = False,
):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
