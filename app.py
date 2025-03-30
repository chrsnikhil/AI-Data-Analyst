from flask import Flask, render_template, request, redirect, url_for, jsonify
from hybrid_search import HybridSearch
import pandas as pd
import os
from mistralai import Mistral
from dotenv import load_dotenv
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()
client = Mistral(api_key="8UnJAKvMMI3ntBLk1FSR94YTjzzsPNCR")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global Variables
product_data = []
hybrid_search = None
columns = []

def process_excel(file_path):
    """Reads Excel file, converts to structured data, and initializes HybridSearch."""
    global product_data, hybrid_search, columns
    df = pd.read_excel(file_path)
    columns = df.columns.tolist()
    product_data = df.to_dict(orient='records')
    hybrid_search = HybridSearch(product_data, columns)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception)),
    before_sleep=lambda retry_state: print(f"Rate limit hit, retrying in {retry_state.next_action.sleep} seconds...")
)
def generate_ai_response(user_message, context):
    """Generates AI response based on user message and data context with rate limit handling."""
    try:
        # Create a context-aware prompt with complete data
        data_str = "\n".join([str(row) for row in product_data])  # Show all rows
        
        prompt = f"""You are a helpful AI assistant analyzing an Excel dataset. The dataset has the following columns: {', '.join(columns)}.
        
        Here's the complete data:
        {data_str}
        
        User message: {user_message}
        
        Please help the user understand the data and find what they're looking for. Be conversational and helpful.
        If the user is searching for something specific, suggest which column might be most relevant.
        If you can see patterns or interesting insights in the data, share them.
        You have access to the complete dataset, so you can provide detailed analysis and insights.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that helps users understand and search through Excel data. You have complete access to the dataset and can provide detailed analysis."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.complete(
            model="mistral-tiny",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,  # Increased token limit for more detailed responses
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred. Please try again. Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, processes Excel file, and initializes search."""
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            process_excel(file_path)
            return redirect(url_for('chat'))
        except Exception as e:
            return str(e)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Handles chat interaction with AI."""
    if request.method == 'POST':
        user_message = request.form.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"})

        # Generate AI response with complete data access
        ai_response = generate_ai_response(user_message, {
            'columns': columns,
            'data': product_data  # Send complete data
        })
        
        return jsonify({
            "response": ai_response,
            "columns": columns
        })
    
    return render_template('chat.html', columns=columns)

@app.route('/search', methods=['POST'])
def search():
    """Handles search requests from the chat interface."""
    search_key = request.form.get('search_key')
    search_by = request.form.get('search_by')
    
    if not search_key or not search_by:
        return jsonify({"error": "Missing search parameters"})
    
    index = hybrid_search.search(search_key, search_by)
    
    if index != -1:
        return jsonify({
            "found": True,
            "data": product_data[index]
        })
    else:
        return jsonify({
            "found": False,
            "message": "No matching records found"
        })

if __name__ == "__main__":
    app.run(debug=True) 