"""
Routes for the OR RAG Bot API.
"""

from flask import Blueprint, request, jsonify
from src.models.optimization_handler import OptimizationHandler

api_bp = Blueprint('api', __name__)

@api_bp.route("/get_formulation", methods=["POST"])
def get_formulation():
    """
    Endpoint to get the mathematical formulation for the given problem.
    Expects a JSON payload with the problem description.
    """
    data = request.json

    opt_handler = OptimizationHandler()

    formulation = opt_handler.format_formulation(data)

    return jsonify({"formulation": formulation})