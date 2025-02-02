from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pathlib import Path
import tiktoken
import os
import json
import logging
from datetime import datetime
from typing import List, Dict

class HierarchicalSummarizer:
    def __init__(
        self,
        model_name: str = "phi4",
        max_tokens_per_chunk: int = 4000,
        min_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        output_folder: str = "summaries"
    ):
        # Setup tokenizer and parameters
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens_per_chunk
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize LLM
        self.llm = ChatOllama(
            model = model_name,
            temperature = 0.8,
            num_predict = 1500,
        )
        
        # Setup text splitter with given parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.encoding.encode(x)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Setup prompt for chunk-level summarization
        self.summary_prompt = PromptTemplate(
            template="""Summarize the following text while preserving key information and insights.

TEXT: {text}

SUMMARY:""",
            input_variables=["text"]
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def read_documents(self, folder_path: str) -> List[str]:
        """Read and return contents of all .txt files in the folder (sorted)."""
        folder = Path(folder_path)
        txt_files = sorted(folder.glob("*.txt"))
        documents = []
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
        return documents

    def merge_documents(self, documents: List[str]) -> str:
        """Merge document texts into a single string."""
        return "\n\n".join(documents)

    def get_optimal_chunk_size(self, text: str) -> int:
        """Determine an optimal chunk size (in characters) based on token limits."""
        total_tokens = len(self.encoding.encode(text))
        if total_tokens <= self.max_tokens:
            return len(text)
        # Estimate characters per token and reserve 90% of max tokens
        chars_per_token = len(text) / total_tokens
        return int(self.max_tokens * chars_per_token * 0.9)

    def split_into_chunks(self, text: str) -> List[str]:
        """Split the text into manageable chunks using the text splitter."""
        docs = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def summarize_chunk(self, chunk_text: str) -> str:
        """Summarize a single chunk using the summary prompt."""
        summary_chain = self.summary_prompt | self.llm
        result = summary_chain.invoke({"text": chunk_text})
        return result.content.strip()

    def save_intermediate_summary(self, summary_text: str, level: int, chunk_index: int) -> None:
        """Save the intermediate summary for a chunk with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_folder / f"level_{level}_chunk_{chunk_index}_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
        self.logger.info(f"Saved summary for level {level} chunk {chunk_index} to {filename}")

    def process_level(self, text: str, level: int) -> List[str]:
        """
        Process one summarization level: split the text into chunks,
        summarize each chunk, save intermediate summaries, and return them.
        """
        self.logger.info(f"Processing summarization level {level}")
        chunks = self.split_into_chunks(text)
        level_summaries = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Summarizing chunk {i+1}/{len(chunks)} at level {level}")
            summary_text = self.summarize_chunk(chunk)
            self.save_intermediate_summary(summary_text, level, i)
            level_summaries.append(summary_text)
        return level_summaries

    def create_final_summary(self, text: str) -> str:
        """Create a final summary from the aggregated summaries."""
        final_prompt = PromptTemplate(
            template="""Create a final summary based on the following section summaries.
Maintain coherence and logical flow while preserving key insights.

SECTION SUMMARIES: {text}

FINAL SUMMARY:""",
            input_variables=["text"]
        )
        final_chain = final_prompt | self.llm
        final_result = final_chain.invoke({"text": text})
        return final_result.content.strip()

    def summarize(self, folder_path: str) -> Dict:
        """
        Main method to perform hierarchical summarization.
        Reads, merges, and processes document content through multiple levels until it fits within limits.
        """
        documents = self.read_documents(folder_path)
        full_text = self.merge_documents(documents)
        metadata = {
            'original_pages': len(documents),
            'original_tokens': len(self.encoding.encode(full_text)),
            'levels': 0
        }
        level = 1
        current_text = full_text
        all_level_summaries = []
        
        # Continue processing until the text is short enough
        while len(self.encoding.encode(current_text)) > self.max_tokens:
            self.logger.info(f"Level {level} summarization starting...")
            level_summaries = self.process_level(current_text, level)
            all_level_summaries.append(level_summaries)
            current_text = "\n\n".join(level_summaries)
            level += 1
        
        metadata['levels'] = level - 1
        final_summary = self.create_final_summary(current_text)
        metadata['final_summary_tokens'] = len(self.encoding.encode(final_summary))
        return {
            'final_summary': final_summary,
            'metadata': metadata,
            'intermediate_summaries': all_level_summaries
        }

def main(folder_path: str = "data/297054", output_folder: str = "data/summaries"):
    os.makedirs(output_folder, exist_ok=True)
    summarizer = HierarchicalSummarizer(output_folder=output_folder)
    result = summarizer.summarize(folder_path)
    if result:
        print("\nFinal Summary:\n", result['final_summary'])
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"{key}: {value}")
        final_summary_path = Path(output_folder) / "final_summary.txt"
        with open(final_summary_path, "w", encoding="utf-8") as f:
            f.write(result['final_summary'])
        print(f"Final summary saved to {final_summary_path}")

if __name__ == "__main__":
    main()
