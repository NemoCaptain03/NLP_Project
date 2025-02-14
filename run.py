import Extract_text
import Summarize_text


def main():
    # Extract text
    pdf_path = "Test.pdf"
    extracted_text_path = "extracted_text.txt"

    text = Extract_text.extract_text_from_pdf(pdf_path)

    # Save extracted text to a file
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Text extraction complete. Saved to", extracted_text_path)

    # Summarize text
    summary_output_path = "summary.txt"
    Summarize_text.summarize_text(extracted_text_path, summary_output_path)

    print("Summarization complete. Saved to", summary_output_path)


if __name__ == "__main__":
    main()
