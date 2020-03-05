from flask_restful import Resource
from flask_restful_swagger_2 import swagger
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('test', help='test field cannot be blank.')

class test(Resource):
    @swagger.doc({
        'tags': ['test'],
        'description': 'Returns a verification note that it works',
        'responses': {
            '200': {
                'description': 'The first api to test the application',
            }
        }
    })

    def get(self):
        return {"responseCode": "SO1",
                "responseMessage":"This stuff works fine"}, 200, \
                {"Access-Control-Allow-Origin": "*"}