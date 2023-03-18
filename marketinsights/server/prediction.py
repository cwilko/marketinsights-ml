from flask_restx import Api, Resource, reqparse, Namespace, fields
from flask import jsonify, request

import os
from marketinsights.server.assembly import PredictionAssembly
from marketinsights.utils.auth import CredentialsStore

api = Api(
    title='MarketInsights Model API',
    version='1.0',
    # All API metadatas
)

ns = Namespace('v1', description='Train, Deploy, and Score models via the MarketInsights API')

api.add_namespace(ns)


@ns.route('/predict/<training_id>')
class Prediction(Resource):

    # model the input data
    model_input = api.model('Provide the features dataset:',
                            {
                                "market": fields.String,
                                "data": fields.List(fields.List(fields.Float)),
                                "tz": fields.String,
                                "index": fields.List(fields.String)
                            })

    parser = reqparse.RequestParser()
    parser.add_argument('market')
    parser.add_argument('data', type=list, action='append')
    parser.add_argument('tz')
    parser.add_argument('index', action='append')

    assembly = PredictionAssembly(CredentialsStore())

    @api.response(200, "Success", model_input)
    @api.expect(model_input)
    def post(self, training_id):
        body = self.parser.parse_args()
        print(body)
        results = self.assembly.score(training_id, body)
        return jsonify(results)
