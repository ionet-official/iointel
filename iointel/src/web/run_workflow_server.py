#!/usr/bin/env python3
"""
Direct entry point for IOIntel workflow server - compatible with Unveil instrumentation.
This creates a direct execution path similar to IRIS api.py, allowing Unveil
to properly instrument the FastAPI app and see workflow executions.
"""
import sys
import webbrowser
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

def open_browser():
    """Open browser after server starts."""
    time.sleep(2)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:8002")
        print(":globe_with_meridians: Opened browser to http://localhost:8002")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

def main():
    """Launch the web interface directly."""
    # Load environment variables
    load_dotenv("creds.env")
    
    print(":rocket: Starting IOIntel WorkflowPlanner Web Interface (Direct Mode)...")
    print(":round_pushpin: Server will be available at: http://localhost:8002")
    print(":wrench: API docs available at: http://localhost:8002/docs")
    print(":black_square_for_stop:  Press Ctrl+C to stop the server")
    print()

    from iointel.src.web.workflow_server import app
    import uvicorn
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    # Run the server directly with the app object
    try:
        uvicorn.run(
            app,  # Direct app object instead of string reference
            host="0.0.0.0",
            port=8002,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n:wave: Server stopped by user")
    except Exception as e:
        print(f":x: Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()