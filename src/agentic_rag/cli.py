"""
Command-line interface for the Agentic RAG system.
"""

import os
import argparse
from typing import List, Optional
import time

from .main import AgenticRAG, initialize_from_data_dir
from .config import DEFAULT_DATA_DIR


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG system for answering queries using local documents and web search."
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing document files for the knowledge base.",
    )
    
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Specific document files to include in the knowledge base.",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process. If provided, the system will answer and exit.",
    )
    
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        help="Specific URLs to scrape when local knowledge is insufficient.",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode, allowing multiple queries.",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Initialize the system
    if args.files:
        print(f"Initializing with specific files: {', '.join(args.files)}")
        rag_system = AgenticRAG(args.files)
    else:
        print(f"Initializing from data directory: {args.data_dir}")
        rag_system = initialize_from_data_dir(args.data_dir)

    # Process a single query and exit
    if args.query:
        start_time = time.time()
        answer = rag_system.process_query(args.query, args.urls)
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print(f"Query: {args.query}")
        print(f"Answer (generated in {elapsed_time:.2f} seconds):")
        print("-" * 50)
        print(answer)
        print("=" * 50)
        return

    # Run in interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to terminate.")
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting interactive mode.")
                break
            
            start_time = time.time()
            try:
                answer = rag_system.process_query(query, args.urls)
                elapsed_time = time.time() - start_time
                
                print("\n" + "=" * 50)
                print(f"Answer (generated in {elapsed_time:.2f} seconds):")
                print("-" * 50)
                print(answer)
                print("=" * 50)
            except Exception as e:
                print(f"Error processing query: {e}")
        
        return
    
    # If no query or interactive mode, show help
    print("No query specified and not in interactive mode. Use --help for usage information.")


if __name__ == "__main__":
    main()