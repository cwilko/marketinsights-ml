from flask import Flask, jsonify 
from flask_restplus import Api, Resource, fields, reqparse 
from flask_cors import CORS 
import os 
from modelClient import MIModelClient
from quantutils.api.bluemix import CloudObjectStore

mc = MIModelClient()

# the app 
app = Flask(__name__) 
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
@ns.route('/predict/<model_id>/<pipeline_id>/<mkt1>/<mkt2>') 
# the endpoint 
class MODEL(Resource): 

    @api.response(200, "Success", model_input)   
    @api.expect(model_input)
    def post(self, model_id, pipeline_id, mkt1, mkt2):        
        body = parser.parse_args()
        model_key = CloudObjectStore.generateKey([model_id, pipeline_id, mkt1, mkt2])
        print(body)
        results = mc.score(model_id, model_key, body)
        return jsonify(results) 

#run
if __name__ == '__main__': 
	app.run(host='0.0.0.0', port=port, debug=False) # deploy with debug=False