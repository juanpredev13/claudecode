# Fixes for RAG Chatbot "Query Failed" Issue

## Root Cause
ChromaDB rejects metadata fields with `None` values. When courses are added to the vector store with `None` values in fields like `instructor`, `course_link`, or `lesson_link`, ChromaDB throws a validation error and the course data is never stored. This results in an empty vector store, causing all queries to return "No relevant content found".

## Test Results Summary
- **63 tests passed** - Core logic is correct
- **4 tests failed** - All due to ChromaDB metadata None values
- **4 errors** - Fixture setup failed due to same issue

## Required Fixes

### Fix 1: vector_store.py - Filter None values from metadata
**Location:** `backend/vector_store.py:135-160`

**Problem:** Passing None values directly to ChromaDB
**Solution:** Filter out None values before adding to ChromaDB

### Fix 2: vector_store.py - Handle None in lesson metadata
**Location:** `backend/vector_store.py:162-180`

**Problem:** CourseChunk metadata may contain None for lesson_number
**Solution:** Ensure metadata only contains non-None values

### Fix 3: document_processor.py - Consider default values
**Location:** `backend/document_processor.py:145`

**Current:** Sets instructor to None
**Consider:** Use empty string "" or "Unknown" instead of None

## Implementation Priority
1. **CRITICAL**: Fix vector_store.py to filter None values (Fixes the immediate issue)
2. **Optional**: Update document_processor.py to avoid None values (Prevents future issues)
