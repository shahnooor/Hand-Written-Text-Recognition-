import os
import cv2
import sqlite3
import matplotlib.pyplot as plt
import easyocr
import os
import cv2
import matplotlib.pyplot as plt
import sqlite3
from language_tool_python import LanguageTool
from nltk.tokenize import sent_tokenize
import spacy
import random
import nltk
nltk.download('punkt')

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")


def ocr_sample(test_data_path):
    # SQLite database setup
    db_path = "output_text.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_results (
            image_name TEXT PRIMARY KEY,
            detected_text TEXT
        )
    ''')
    conn.commit()

    # Use EasyOCR to perform text recognition with multiple languages
    reader = easyocr.Reader(['en'], gpu=True)

    # LanguageTool for grammar and spelling correction
    tool = LanguageTool('en-US')

    # Get a list of all pre-processed image files in the folder
    preprocessed_files = [f for f in os.listdir(test_data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process a random set of 3 images
    random_images = random.sample(preprocessed_files, min(10, len(preprocessed_files)))

    for i, image_name in enumerate(random_images):
        # Read the pre-processed image
        preprocessed_path = os.path.join(test_data_path, image_name)

        # Load the image
        preprocessed_img = cv2.imread(preprocessed_path)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to improve contrast
        _, threshold_img = cv2.threshold(gray_img, 0, 260, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply GaussianBlur to reduce noise
        blurred_img = cv2.GaussianBlur(threshold_img, (5, 5), 0)

        # Resize the image to a standard size
        resized_img = cv2.resize(blurred_img, (800, 600))

        # Perform OCR on the pre-processed image
        result = reader.readtext(resized_img)

        # Extract all text without filtering
        detected_text = ' '.join([detection[1] for detection in result])

        # Use language_tool_python for grammar and spelling correction
        corrected_text = tool.correct(detected_text)

        # Apply spaCy for lemmatization and part-of-speech tagging
        doc = nlp(corrected_text)
        improved_text = ' '.join([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])

        # Tokenize into sentences for better formatting
        sentences = sent_tokenize(improved_text)

        # Insert results into SQLite database
        cursor.execute("INSERT OR REPLACE INTO text_results (image_name, detected_text) VALUES (?, ?)", (image_name, improved_text))
        conn.commit()

        # Display the pre-processed image with bounding boxes around detected text
        img_with_boxes = resized_img.copy()
        for detection in result:
            box = detection[0]
            text = detection[1]
            score = detection[2]

            (top_left, top_right, bottom_right, bottom_left) = box

            # Highlight all text with green bounding boxes
            img_with_boxes = cv2.rectangle(img_with_boxes, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
            img_with_boxes = cv2.putText(img_with_boxes, f"{text} ({score:.2f})", tuple(map(int, top_left)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the pre-processed image with bounding boxes
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Text (EasyOCR) - {image_name}")
        plt.axis('off')
        plt.show()

        # Display the corrected text as a paragraph
        print(f"Detected Text ({image_name}):")
        for sentence in sentences:
            print(sentence)
        print("\n---\n")

    # Close SQLite connection
    conn.close()
