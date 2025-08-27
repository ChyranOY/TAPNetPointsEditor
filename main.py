#!/usr/bin/env python3
"""
TAPNet Tracker - Main Entry Point

A comprehensive video tracking solution using TAPNext model.
This script provides the main entry point to run the Gradio web interface.

Usage:
    python main.py [--port PORT] [--host HOST] [--share]

Examples:
    python main.py                    # Run on localhost:7860
    python main.py --port 8080        # Run on localhost:8080
    python main.py --share            # Create public link
"""

import argparse
import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tapnet_tracker import create_tapnext_app
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are properly installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="TAPNet Tracker - Video trajectory tracking tool based on TAPNext model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py                    # Run on localhost:7860
  python main.py --port 8080        # Run on localhost:8080  
  python main.py --share            # Create public sharing link
  python main.py --host 0.0.0.0     # Allow external access
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Server port number (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1", 
        help="Server host address (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link (via gradio.live)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting TAPNet Tracker...")
    print("=" * 50)
    print("🎯 TAPNext Trajectory Generator")
    print("📝 Supports video trajectory generation, visualization and interactive editing")
    print("=" * 50)
    
    try:
        # Create Gradio application
        print("📦 Creating application interface...")
        app = create_tapnext_app()
        
        # Start server
        print(f"🌐 Starting server...")
        print(f"   Address: http://{args.host}:{args.port}")
        if args.share:
            print("🔗 Will create public sharing link...")
        
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 User interrupted, shutting down server...")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 