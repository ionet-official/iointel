#!/usr/bin/env python3
"""
Launch the WorkflowPlanner web interface.

This script starts a FastAPI server that serves the React Flow web interface
for creating and visualizing workflows.
"""

import sys
import webbrowser
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Launch the web interface."""
    # Load environment variables
    load_dotenv("creds.env")
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("🚀 Starting WorkflowPlanner Web Interface...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔧 API docs available at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    # Import and run the server
    try:
        import uvicorn
        from iointel.src.web.workflow_server import app
        
        # Open browser after a short delay
        def open_browser():
            import time
            time.sleep(2)  # Wait for server to start
            try:
                webbrowser.open("http://localhost:8000")
                print("🌐 Opened browser to http://localhost:8000")
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
        
        # Start browser opening in background
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()