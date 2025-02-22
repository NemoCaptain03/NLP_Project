# Lecture Slides Summarization Tool

## Overview
This project extracts text from PDF files and summarizes it using a transformer-based language model. The system utilizes **PyPDF2** for text extraction and **DistilBART-CNN** for summarization. The extracted and summarized text is saved for further use.

## Project Structure
```
|-- Extract.py         # Handles text extraction from PDF
|-- Summarize.py       # Summarizes extracted text
|-- main.py            # Integrates extraction and summarization
|-- requirements.txt   # Dependencies for the project
|-- README.md          # Project documentation
|-- extract.txt        # Extracted text (generated during execution)
|-- summary.txt        # Summarized output (generated during execution)
```

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.10+** installed.

### Install Dependencies
Run the following command to install required packages:
```sh
pip install -r requirements.txt
```

## Running the System
Execute the `main.py` file to extract text and summarize it:
```sh
python main.py
```
The input PDF file should be placed in the project directory and called in **pdf_path** in `main.py`. The extracted text will be saved in `extract.txt`, and the summarized text in `summary.txt`.

## Data & Model
### Downloading the Model
This project uses **DistilBART-CNN** for text summarization. It is automatically downloaded from Hugging Face when first executed.
- Model: `sshleifer/distilbart-cnn-12-6`
- Tokenizer: `AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")`

### Input Data
- Place the PDF file in the project directory and ensure the name in **pdf_path** before running `main.py`.

## References
This project is inspired by:
- Hugging Face tutorials for summarization: https://youtu.be/CDmPBsZ09wg?si=SdlW1nKAIgdt7VbO
