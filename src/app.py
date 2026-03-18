"""
Flask web application for the OR RAG chatbot.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

from src.models.embeddings import EmbeddingHandler
from src.models.vector_store import VectorStore
from src.models.llm_handler import LLMHandler
from src.utils.solver_integration import SolverIntegration
from src.config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

embedding_handler = None
vector_store = None
llm_handler = None
solver_integration = None

def initialize_components():
    """Initialize all RAG components."""
    global embedding_handler, vector_store, llm_handler, solver_integration
    
    logger.info("Initializing RAG components...")
    
    try:
        embedding_handler = EmbeddingHandler()
        vector_store = VectorStore()
        llm_handler = LLMHandler()
        solver_integration = SolverIntegration()
        
        logger.info("All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """
    Handle user queries.
    
    Expected JSON:
    {
        "query": "User's question",
        "query_type": "formulation|sensitivity|code|explanation",
        "context": "Optional additional context"
    }
    """
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        query_type = data.get('query_type', 'formulation')
        additional_context = data.get('context', '')
        
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
            
        logger.info(f"Received query: {user_query[:100]}... (type: {query_type})")
        
        query_embedding = embedding_handler.embed_text(user_query)
        
        results = vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=5
        )
        
        context_text = "\n\n".join([
            f"[Retrieved Document {i+1}]\n{doc}"
            for i, doc in enumerate(results['documents'])
        ])
        
        if query_type == 'formulation':
            response = llm_handler.formulate_problem(user_query, context_text)
        elif query_type == 'sensitivity':
            response = llm_handler.analyze_sensitivity(
                additional_context, user_query, context_text
            )
        elif query_type == 'code':
            response = llm_handler.generate_code(user_query, context_text)
        elif query_type == 'explanation':
            response = llm_handler.explain_concept(user_query, context_text)
        else:
            response = llm_handler.generate_response(user_query, context_text)
            
        code_output = None
        if '```python' in response:
            code = solver_integration.extract_code_from_response(response)
            if code:
                success, output, results_dict = solver_integration.execute_code(code)
                if success:
                    code_output = output
                    
        return jsonify({
            'response': response,
            'code_output': code_output,
            'retrieved_docs_count': len(results['documents']),
            'similarity_scores': results['distances'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics."""
    try:
        vector_stats = vector_store.get_collection_stats()
        embedding_info = embedding_handler.get_model_info()
        
        return jsonify({
            'vector_db': vector_stats,
            'embedding_model': embedding_info,
            'llm_model': llm_handler.model_name,
            'status': 'operational'
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter', methods=['POST'])
def filter_query():
    """
    Query with metadata filters.
    
    Expected JSON:
    {
        "query": "User's question",
        "filters": {
            "problem_type": "LP",
            "industry": "Manufacturing"
        }
    }
    """
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        filters = data.get('filters', {})
        
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
            
        query_embedding = embedding_handler.embed_text(user_query)
        
        results = vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=5,
            filter_metadata=filters
        )
        
        return jsonify({
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'distances': results['distances'],
            'count': len(results['documents'])
        })
        
    except Exception as e:
        logger.error(f"Error in filtered query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting OR RAG Bot Flask Application...")
    
    if not initialize_components():
        logger.error("Failed to initialize. Please run src/main.py first to build the knowledge base.")
        exit(1)
        
    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)