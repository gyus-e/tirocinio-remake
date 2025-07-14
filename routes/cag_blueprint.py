from flask import Blueprint, request, jsonify
from models import Configuration

cag_blueprint = Blueprint("cag", __name__)


@cag_blueprint.post("/<config_id>/chat")
async def cag_chat(config_id):
    
    config: Configuration | None = Configuration.query.filter_by(id=config_id).first()
    if not config:
        return jsonify(message="Configuration not found"), 404

    try:
        query = validate_cag_chat_request(request)
    except ValueError as e:
        return jsonify(message=str(e)), 400

    answer = "PLACEHOLDER"

    return jsonify(answer=answer), 200


def validate_cag_chat_request(request) -> str:
    data = request.get_json() if request.is_json else request.form
    if not data:
        raise ValueError("Invalid request: No data provided")

    query = data.get("query")
    if not query:
        raise ValueError("Invalid request: No query provided")

    return query