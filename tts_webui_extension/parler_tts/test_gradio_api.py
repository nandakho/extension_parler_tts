#!/usr/bin/env python3
"""
Integration test for Parler TTS Gradio API endpoint.
This test starts the Parler TTS extension as a Gradio app and tests the API endpoints.
"""

import sys
import os
import time
import subprocess
import socket

def test_gradio_api():
    """Test the Parler TTS Gradio API by starting the app and calling it via client."""
    gradio_process = None
    try:
        # Start the Gradio app by running python demo.py from the extension directory
        extension_dir = os.path.dirname(__file__)
        gradio_process = subprocess.Popen(
            [sys.executable, "demo.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=extension_dir
        )

        # Wait for the server to start (check if port 7860 is listening - default Gradio port)
        max_attempts = 20  # Parler TTS might take longer to load
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 7860))
                sock.close()
                if result == 0:
                    print("✓ Gradio server started successfully")
                    break
            except:
                pass
            time.sleep(3)  # Wait longer for Parler TTS to load
        else:
            raise Exception("Gradio server failed to start within timeout")

        # Now try to connect with Gradio client
        from gradio_client import Client

        client = Client("http://127.0.0.1:7860")

        # Test the parler_tts API endpoint
        print("Testing parler_tts API endpoint...")
        result = client.predict(
            text="Hello, this is a test.",
            description="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
            model_name="parler-tts/parler-tts-mini-v1",
            attn_implementation="eager",
            compile_mode=None,
            seed=42,
            api_name="/parler_tts"
        )

        print("✓ Parler TTS API call executed successfully")
        print(f"✓ Result type: {type(result)}")

        # The result should be a dict with audio_out and other metadata
        if isinstance(result, dict):
            print(f"✓ Result keys: {list(result.keys())}")
            if "audio_out" in result:
                audio_data = result["audio_out"]
                print(f"✓ Audio data received: {type(audio_data)}")
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    sample_rate, audio_array = audio_data
                    print(f"✓ Sample rate: {sample_rate}, Audio length: {len(audio_array)}")
                else:
                    print("⚠ Audio data format unexpected")
            else:
                print("⚠ No audio_out in result")
        else:
            print("⚠ Result is not a dict")

        return True

    except Exception as e:
        print(f"✗ Gradio API test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Stop the gradio process
        if gradio_process:
            try:
                gradio_process.terminate()
                gradio_process.wait(timeout=15)  # Give more time for Parler TTS to cleanup
            except:
                try:
                    gradio_process.kill()
                except:
                    pass


def check_gradio_server_running():
    """Check if the Gradio server is running on port 7860."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 7860))
        sock.close()
        return result == 0
    except:
        return False


if __name__ == "__main__":
    print("Parler TTS Gradio API Integration Test")
    print("=" * 50)

    # Check if server is running
    if not check_gradio_server_running():
        print("Starting Parler TTS Gradio server...")
        print("Note: This test will start the Parler TTS extension as a Gradio app.")
        print("Make sure you have the required dependencies installed.")
        print("This may take a while to load the model.")
        print()

    # Run the test
    success = test_gradio_api()

    print("\n" + "=" * 50)
    if success:
        print("✓ Integration test PASSED - Parler TTS Gradio API is working!")
        sys.exit(0)
    else:
        print("✗ Integration test FAILED - Check the logs for more details")
        sys.exit(1)
