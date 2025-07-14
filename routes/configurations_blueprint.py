from flask import Blueprint, request, jsonify
from models import Configuration
from utils import DB
from utils.validators import validate_configuration_request

configurations_blueprint = Blueprint("configurations", __name__)


@configurations_blueprint.get("/configurations")
def get_configurations():
    return Configuration.query.all(), 200


@configurations_blueprint.get("configurations/<config_id>")
def get_configuration(config_id):
    configuration = Configuration.query.filter_by(id=config_id).first()
    if not configuration:
        return jsonify(message="Configuration not found"), 404
    return configuration, 200


@configurations_blueprint.post("/configurations")
def initialize_configuration():
    try:
        system_prompt, model_name, rag_configuration, warnings = validate_configuration_request(request)
        configuration = Configuration(
            system_prompt=system_prompt, 
            model_name=model_name,
            rag_configuration=rag_configuration
            )
        DB.session.add(configuration)
        DB.session.commit()

    except ValueError as e:
        return jsonify(message=str(e)), 400

    return jsonify(
        configuration_id=configuration.config_id,
        warnings=warnings if warnings else None,
        ), 200
