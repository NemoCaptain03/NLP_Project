import Extract
import Summarize


def main():
    # Extract text
    pdf_path = "Introduction to NLP.pdf"
    extracted_text_path = "extract.txt"

    text = Extract.extract_text_from_pdf(pdf_path)

    # Save extracted text to a file
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Text extraction complete. Saved to", extracted_text_path)

    # Summarize text
    summary_output_path = "summary.txt"
    Summarize.summarize_text(extracted_text_path, summary_output_path)


if __name__ == "__main__":
    main()
