from flask import Flask, request, render_template
import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time

# Initialize Flask app and T5 tokenizer/model
app = Flask(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Use smaller model for speed
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)  # Using smaller model

# Function to summarize text into a paragraph
def summarize_text_paragraph(text, max_length=300, min_length=100):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=2, early_stopping=True)  # Reduced beams
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Improved chunking function to ensure efficient tokenization
def chunk_text(text, max_tokens=512):
    sentences = text.split('. ')  # Split the text into sentences
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the sentence exceeds max token limit
        if len(tokenizer.encode(current_chunk + sentence)) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            if current_chunk:  # Avoid adding empty chunks
                chunks.append(current_chunk)
            current_chunk = sentence + ". "  # Start a new chunk

    if current_chunk:  # Add the last chunk if it isn't empty
        chunks.append(current_chunk)
    
    return chunks

# Function to extract text by page from the PDF
def extract_text_by_page(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    return pages

# Batch summarization function
def summarize_text_batch(texts, max_length=300, min_length=100):
    inputs = tokenizer.batch_encode_plus(
        ["summarize: " + text for text in texts],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=1024
    ).to(device)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=2,  # Reduced beams
        early_stopping=True
    )
    
    return [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        if pdf_file:
            # Extract text from each page of the uploaded PDF
            start_time = time.time()  # Track time for performance testing
            pages = extract_text_by_page(pdf_file)
            summaries = []

            # Process each page
            for page in pages:
                if page:
                    # First, chunk the page if the content is too large
                    chunks = chunk_text(page)
                    # Summarize multiple chunks at once
                    page_summary = summarize_text_batch(chunks)
                    
                    summaries.append(' '.join(page_summary))

            end_time = time.time()
            print(f"Processing Time: {end_time - start_time} seconds")

            return render_template('index.html', summaries=summaries)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
