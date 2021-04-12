import json
from flask import Flask, jsonify, request
from typing import Optional
from qanta.model_proxy import guess_and_buzz, batch_guess_and_buzz, ModelProxy
from typing import Optional
from typing import Tuple, List


def create_app(config_file: str, enable_batch: Optional[bool] = True):
    '''
    Http interface for the model
    Args:
        config_file: str, yaml config file of the model used for online service
        enable_batch: bool, if batch post request is allowed
    '''

    guesser = ModelProxy.load(config_file)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(guesser, questions)
        ])

    return app
