# PASSKEY: rushit2712
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

import torch

if has_spaces:
    @spaces.GPU
    def dummy_gpu_fn():
        return torch.cuda.is_available()
else:
    def dummy_gpu_fn():
        return torch.cuda.is_available()

import os
import sys
import subprocess


# 1. Configure local user-space compilation paths for TA-Lib
home_dir = os.path.expanduser("~")
ta_lib_prefix = os.path.join(home_dir, "ta-lib")
ta_lib_lib = os.path.join(ta_lib_prefix, "lib")
ta_lib_include = os.path.join(ta_lib_prefix, "include")

# Export variables so that the compiler and python loader look inside this local directory
os.environ["TA_LIBRARY_PATH"] = ta_lib_lib
os.environ["TA_INCLUDE_PATH"] = ta_lib_include
os.environ["LD_LIBRARY_PATH"] = f"{ta_lib_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["LIBRARY_PATH"] = f"{ta_lib_lib}:{os.environ.get('LIBRARY_PATH', '')}"
os.environ["C_INCLUDE_PATH"] = f"{ta_lib_include}:{os.environ.get('C_INCLUDE_PATH', '')}"

# 2. Compile C library to local prefix if missing
if not os.path.exists(os.path.join(ta_lib_lib, "libta_lib.so")):
    print("[TA-Lib Compiler] Local TA-Lib C library missing. Compiling from source...")
    try:
        os.makedirs(ta_lib_prefix, exist_ok=True)
        # Download TA-Lib C source
        subprocess.run(["wget", "-q", "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"], check=True)
        # Extract source
        subprocess.run(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"], check=True)
        # Configure, build and install locally (in user space, bypassing root requirements)
        subprocess.run([
            "sh", "-c", 
            f"cd ta-lib && ./configure --prefix={ta_lib_prefix} && make -s && make install -s"
        ], check=True)
        # Cleanup source files
        subprocess.run(["rm", "-rf", "ta-lib", "ta-lib-0.4.0-src.tar.gz"])
        print("[TA-Lib Compiler] Local compilation and installation complete!")
    except Exception as e:
        print(f"[TA-Lib Compiler] Error compiling C library: {e}")

# 3. Compile/Install Python TA-Lib package wrapper dynamically
try:
    import talib
except ImportError:
    print("[TA-Lib Wrapper] Python library missing. Installing dynamically from PyPI...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-build-isolation", "TA-Lib==0.6.8"
        ], check=True)
        import talib
        print("[TA-Lib Wrapper] Python wrapper compiled and imported successfully!")
    except Exception as e:
        print(f"[TA-Lib Wrapper] Failed to install Python library wrapper: {e}")

# 4. Now import the main FastAPI trading console (with optional Gradio wrapper for ZeroGPU)
try:
    import gradio as gr
    from backend_engine.web_app import app as fastapi_app
    with gr.Blocks(title="Algo Trading Console") as demo:
        gr.Markdown("# ⚡ Algorithmic Trading Console Active")
        gr.Markdown("The main trading console is running on the root path `/`.")
        btn = gr.Button("Check GPU Node Status")
        out = gr.Textbox(label="Status")
        btn.click(fn=dummy_gpu_fn, outputs=out)
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
except ImportError:
    from backend_engine.web_app import app as fastapi_app
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
