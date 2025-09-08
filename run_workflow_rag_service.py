#!/usr/bin/env python3
"""
Run the Workflow RAG Service
"""

from iointel.src.web.workflow_rag_service import run_server

if __name__ == "__main__":
    # Run on port 8101 to avoid conflicts with other services
    run_server(host="0.0.0.0", port=8101)