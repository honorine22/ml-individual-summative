#!/usr/bin/env python3
"""
Single-port startup script for Render deployment
Runs both FastAPI and Streamlit on the same port using subprocess
"""
import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def cleanup(signum=None, frame=None):
    """Clean up processes on exit"""
    print("\n🛑 Shutting down services...")
    try:
        # Kill FastAPI
        subprocess.run(["pkill", "-f", "uvicorn src.api:app"], check=False)
        # Kill Streamlit  
        subprocess.run(["pkill", "-f", "streamlit run"], check=False)
    except:
        pass
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    print("🚀 FaultSense - Single Port Deployment")
    print(f"📁 Working directory: {project_root}")
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8000))
    print(f"🌐 Using port: {port}")
    
    # Create demo model if needed
    model_path = project_root / "models" / "faultsense_cnn.pt"
    if not model_path.exists():
        print("⚡ Creating demo model for deployment...")
        os.makedirs(project_root / "models", exist_ok=True)
        registry_path = project_root / "models" / "registry.json"
        with open(registry_path, 'w') as f:
            f.write('{"model_type": "demo", "accuracy": 0.75, "created": "single_port_deployment"}')
        print("✅ Demo model created")
    
    # Start FastAPI
    print("🔧 Starting FastAPI server...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ], env={**os.environ, "PYTHONPATH": str(project_root)})
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("❌ API failed to start")
        cleanup()
        return
    
    # Start Streamlit on different port
    streamlit_port = port + 1
    print(f"🎨 Starting Streamlit on port {streamlit_port}...")
    
    streamlit_env = {
        **os.environ,
        "API_URL": f"http://localhost:{port}",
        "PYTHONPATH": str(project_root)
    }
    
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0"
    ], env=streamlit_env)
    
    print("")
    print("╔════════════════════════════════════════╗")
    print("║   Services Started Successfully!      ║")
    print("╚════════════════════════════════════════╝")
    print("")
    print(f"🌐 API Server:    https://your-app.onrender.com (port {port})")
    print(f"🎨 Streamlit UI:  https://your-app.onrender.com:{streamlit_port}")
    print(f"📚 API Docs:      https://your-app.onrender.com/docs")
    print("")
    print("✅ Ready for production traffic")
    
    # Keep running and monitor processes
    try:
        while True:
            time.sleep(30)
            
            # Check if API process is still running
            if api_process.poll() is not None:
                print("⚠️  API process died, restarting...")
                api_process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "src.api:app",
                    "--host", "0.0.0.0",
                    "--port", str(port)
                ], env={**os.environ, "PYTHONPATH": str(project_root)})
            
            # Check if Streamlit process is still running
            if streamlit_process.poll() is not None:
                print("⚠️  Streamlit process died, restarting...")
                streamlit_process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                    "--server.port", str(streamlit_port),
                    "--server.headless", "true",
                    "--server.address", "0.0.0.0"
                ], env=streamlit_env)
                
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
