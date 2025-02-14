from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download sentence tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# Load model and tokenizer
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


def summarize_text(file_path, output_path):
    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Chunk the text into manageable sizes
    length = 0
    chunk = ""
    chunks = []
    count = -1

    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length

        if combined_length <= tokenizer.max_len_single_sentence:
            chunk += sentence + " "
            length = combined_length

            if count == len(sentences) - 1:
                chunks.append(chunk.strip())
        else:
            chunks.append(chunk.strip())
            length = 0
            chunk = sentence + " "
            length = len(tokenizer.tokenize(sentence))

    # Summarize each chunk
    summarized_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        output = model.generate(**inputs, max_length=30, min_length=10, length_penalty=2.0, do_sample=False)
        summarized_text = tokenizer.decode(output[0], skip_special_tokens=True)
        summarized_chunks.append(summarized_text)

    # Save summarized text
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summarized_chunks))

    print("Summarization complete. Saved to", output_path)