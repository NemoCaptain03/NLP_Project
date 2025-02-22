from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download sentence tokenizer
nltk.download('punkt')

# Load model and tokenizer
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Maximum token length for the model
MAX_TOKENS = 1024


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS):
    """Splits text into chunks that fit within the token limit."""
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_length = len(tokenizer.tokenize(sentence))

        if current_length + tokenized_length <= max_tokens - 50:  # Reserve space for special tokens
            current_chunk.append(sentence)
            current_length += tokenized_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = tokenized_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_chunk(chunk, tokenizer, model):
    """Summarizes a single text chunk."""
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)

    summary_ids = model.generate(
        **inputs,
        max_length=min(len(chunk) // 2, 512),
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        repetition_penalty=1.5,
        do_sample=False
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_text(file_path, output_path):
    """Reads, processes, summarizes, and saves the text."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Step 1: Chunk text properly
    chunks = chunk_text(text, tokenizer, max_tokens=MAX_TOKENS)

    # Step 2: Summarize each chunk
    summarized_chunks = [summarize_chunk(chunk, tokenizer, model) for chunk in chunks]

    # Step 3: Save summarized text
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(summarized_chunks))

    print(f"Summarization complete. Saved to {output_path}")

