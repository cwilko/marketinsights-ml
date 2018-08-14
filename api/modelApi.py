from flask import Flask, jsonify 
from flask_restplus import Api, Resource, fields, reqparse 
from flask_cors import CORS 
from flask_sslify import SSLify
import os 
from quantutils.model.mimodelclient import MIModelClient
from quantutils.api.auth import CredentialsStore

mc = MIModelClient(CredentialsStore('api/cred'))

# the app 
app = Flask(__name__) 
SSLify(app)
CORS(app) 
api = Api(app, version='1.0', title='MarketInsights Model API', validate=False) 
ns = api.namespace('marketinsights', 'Train, Deploy, and Score models via the MarketInsights API') 
# load the algo 

# model the input data 
model_input = api.model('Provide the features dataset:', 
	{ 
		"market": fields.String,
		"data": fields.List(fields.List(fields.Float)),
		"tz": fields.String,
		"index": fields.List(fields.String)
	})

# the input data type here is Integer. You can change this to whatever works for your app. 
# On Bluemix, get the port number from the environment variable PORT # When running this app on the local machine, default to 8080 
port = int(os.getenv('PORT', 8080)) 
parser = reqparse.RequestParser()
parser.add_argument('market')
parser.add_argument('data', type=list, action='append')
parser.add_argument('tz')
parser.add_argument('index', action='append')

# The ENDPOINT 
@ns.route('/predict/<training_id>') 
# the endpoint 
class MODEL(Resource): 

    @api.response(200, "Success", model_input)   
    @api.expect(model_input)
    def post(self, training_id):        
        body = parser.parse_args()
        print(body)
        results = mc.score(training_id, body)
        return jsonify(results) 

#run
if __name__ == '__main__': 
	app.run(host='0.0.0.0', port=port, debug=False) # deploy with debug=False