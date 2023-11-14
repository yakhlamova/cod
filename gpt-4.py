import os
from dotenv import load_dotenv

import PyPDF2
import unicodedata
import openai

from typing import List
from dataclasses import dataclass
from llm_core.assistants import OpenAIAssistant


# Load environment variables from .env
load_dotenv()

# Retrieve the API key from environment variables Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to clean up unicode characters from the PDF
def cleanup_unicode(text):
    corrected_chars = []
    for char in text:
        corrected_char = unicodedata.normalize("NFKC", char)
        corrected_chars.append(corrected_char)
    return "".join(corrected_chars)


# Open the PDF file and extract text
with open(
    "Why polar bears are no longer the poster image of climate change - BBC Future.pdf",
    "rb",
) as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page in pdf_reader.pages:
        pages.append(page.extract_text())
    text = "".join(pages)

# Cleanup unicode characters
text = cleanup_unicode(text)


# Define data classes for summarization
@dataclass
class DenseSummary:
    denser_summary: str
    missing_entities: List[str]


@dataclass
class DenserSummaryCollection:
    summaries: List[DenseSummary]

    @classmethod
    def summarize(cls, article):
        with OpenAIAssistant(cls, model="gpt-4") as assistant:
            return assistant.process(article=article)


# Create and print the summaries
summary_collection = DenserSummaryCollection.summarize(text)
print(len(summary_collection.summaries))
print(summary_collection.summaries[0].missing_entities)
print(summary_collection.summaries[0].denser_summary)
print(summary_collection.summaries[1].missing_entities)
print(summary_collection.summaries[1].denser_summary)
