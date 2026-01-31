#!/usr/bin/env python3
"""
Upload course files to OpenAI and create a vector store for RAG.

Usage:
    make upload        # Upload files and create vector store
    make list-files    # List files in the vector store

After running, copy the VECTOR_STORE_ID to config.ini.

Note: OpenAI doesn't support .typ files, so they are converted to .txt
with a header explaining the format.
"""

import os
import sys
import tempfile
import shutil
import configparser
from openai import OpenAI

from chatgpt import load_course_materials, match_file_pattern


def upload_course_files(
    api_key: str,
    course_dir: str,
    file_patterns: list,
    exclude_patterns: list = None,
    vector_store_name: str = "DSAA3071 Course Materials"
) -> str:
    """
    Upload course files to OpenAI and create a vector store.
    
    Args:
        api_key: OpenAI API key
        course_dir: Path to course materials directory
        file_patterns: Patterns to include (e.g., ["*learning-sheet*", "*validation*"])
        exclude_patterns: Patterns to exclude (e.g., ["*test*"])
        vector_store_name: Name for the vector store
        
    Returns:
        Vector store ID
    """
    client = OpenAI(api_key=api_key)
    
    # Load all materials
    print(f"Loading materials from: {course_dir}")
    materials = load_course_materials(course_dir)
    
    if not materials:
        print("ERROR: No materials found!")
        sys.exit(1)
    
    # Collect matching files with their content
    exclude_patterns = exclude_patterns or []
    files_to_upload = []
    
    for week, files in sorted(materials.items()):
        for rel_path, content in sorted(files.items()):
            # Check include patterns
            if not match_file_pattern(rel_path, file_patterns):
                continue
            
            # Check exclude patterns
            if any(match_file_pattern(rel_path, [ep]) for ep in exclude_patterns):
                print(f"  Skipping (excluded): {rel_path}")
                continue
            
            files_to_upload.append((rel_path, content))
    
    if not files_to_upload:
        print("ERROR: No files match the patterns!")
        sys.exit(1)
    
    print(f"\nFiles to upload ({len(files_to_upload)}):")
    for rel_path, _ in files_to_upload:
        print(f"  - {rel_path}")
    
    # Create temp directory for converted files
    temp_dir = tempfile.mkdtemp(prefix="dsaa3071_upload_")
    print(f"\nConverting files to .txt format in: {temp_dir}")
    
    try:
        # Upload files
        print("\nUploading files to OpenAI...")
        file_ids = []
        
        for rel_path, content in files_to_upload:
            # Convert .typ to .txt (OpenAI doesn't support .typ)
            txt_filename = rel_path.replace("/", "_").replace(".typ", ".txt")
            txt_path = os.path.join(temp_dir, txt_filename)
            
            # Add header to explain the content
            header = f"""# {rel_path}
# DSAA3071 Theory of Computation - Course Material
# Format: Typst (similar to LaTeX/Markdown)
# ============================================

"""
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(header + content)
            
            print(f"  Uploading: {rel_path} -> {txt_filename}...", end=" ", flush=True)
            try:
                with open(txt_path, "rb") as f:
                    file_obj = client.files.create(
                        file=(txt_filename, f, "text/plain"),
                        purpose="assistants"  # Required for file_search
                    )
                file_ids.append(file_obj.id)
                print(f"OK ({file_obj.id})")
            except Exception as e:
                print(f"FAILED: {e}")
    
        if not file_ids:
            print("ERROR: No files uploaded successfully!")
            sys.exit(1)
        
        print(f"\nUploaded {len(file_ids)} files successfully.")
        
        # Create vector store
        print(f"\nCreating vector store: {vector_store_name}...")
        vector_store = client.vector_stores.create(
            name=vector_store_name,
            file_ids=file_ids
        )
        
        print(f"Vector store created: {vector_store.id}")
        print(f"Status: {vector_store.status}")
        
        # Wait for processing
        print("\nWaiting for files to be processed...")
        import time
        while True:
            vs = client.vector_stores.retrieve(vector_store.id)
            completed = vs.file_counts.completed
            total = vs.file_counts.total
            in_progress = vs.file_counts.in_progress
            
            print(f"  Progress: {completed}/{total} completed, {in_progress} in progress")
            
            if vs.status == "completed" or in_progress == 0:
                break
            
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print("SUCCESS! Add this to your config.ini:")
        print(f"VECTOR_STORE_ID = {vector_store.id}")
        print(f"{'='*60}")
        
        return vector_store.id
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    # Read configuration
    config = configparser.ConfigParser()
    config.read("config.ini")
    settings = config["settings"]
    
    api_key = settings["OPENAI_API_KEY"]
    course_dir = settings.get("COURSE_DIR", "DSAA3071TheoryOfComputation")
    
    # File patterns - include learning sheets and validation, exclude tests
    file_patterns_str = settings.get("FILE_PATTERNS", "*learning-sheet*, *validation*")
    file_patterns = [p.strip() for p in file_patterns_str.split(",") if p.strip()]
    
    # Always exclude test files
    exclude_patterns = ["*test*", "*test.B*"]
    
    print("DSAA3071 Course Materials - OpenAI Upload")
    print("=" * 60)
    print(f"Course directory: {course_dir}")
    print(f"Include patterns: {file_patterns}")
    print(f"Exclude patterns: {exclude_patterns}")
    print()
    
    upload_course_files(
        api_key=api_key,
        course_dir=course_dir,
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns,
        vector_store_name="DSAA3071 Theory of Computation"
    )


if __name__ == "__main__":
    main()
