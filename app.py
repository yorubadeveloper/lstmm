from flask import Flask
from flask_cors import CORS
from flask_restful_swagger_2 import Api
from flask_swagger_ui import get_swaggerui_blueprint
from flask import redirect
import resources as rn

SWAGGER_URL = '/api/docs'

API_URL = '/api/swagger.json'

app = Flask(__name__)
api = Api(app, api_version='1.0', title="LSTMM-NN")
CORS(app)

api.add_resource(rn.test, '/api/test')


swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "LSTM-NN"
    }

)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/")
def main():
    return redirect("/api/docs")


if __name__ == "__main__":
    app.run()
