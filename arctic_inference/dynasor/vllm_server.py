import asyncio
import vllm
import requests
import time
import multiprocessing
import signal
import sys
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.utils import cli_env_setup
import arctic_inference.dynasor.openai_server as openai_server

def check_health(port: int, retry_interval: float = 1.0) -> bool:
    """Check if the server is healthy by querying the /health endpoint."""
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(retry_interval)


def run_vllm_server(args):
    """Run the vLLM server in a separate process."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_server(args))
    return


def run_openai_server(config):
    """Run the OpenAI server in a separate process."""
    openai_server.set_config(config)
    openai_server.start_server(config)
    return


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

def main():
    
    register_signal_handlers()

    parser = FlexibleArgumentParser(
        description="Arctic Inference Dynasor Proxy w/ vLLM Server"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    vllm_port = args.port
    openai_port = vllm_port
    args.port = vllm_port + 1
    vllm_port = args.port
    validate_parsed_serve_args(args)

    # Start the server in a separate process
    server_process = multiprocessing.Process(target=run_vllm_server, args=(args,))

    
    openai_config = openai_server.parse_args([
        "--port", str(openai_port),
        "--target-base-url", f"http://localhost:{vllm_port}",
    ])
    print(f"Starting OpenAI server: {openai_config}")
    proxy_server = multiprocessing.Process(
        target=run_openai_server, 
        args=(openai_config,)
    )
    



    server_process.start()

    # Wait for the server to be healthy
    if check_health(vllm_port):
        print(f"Server is healthy and running on port {args.port}")
    else:
        print("Server failed to become healthy within the timeout period")
        server_process.terminate()
        server_process.join()
        sys.exit(1)

    proxy_server.start()

    # Wait for the OpenAI server to be healthy
    if check_health(openai_port):
        print(f"Dynasor Proxy server is healthy and running on port {openai_port}")
    else:
        print("Dynasor Proxy server failed to become healthy within the timeout period")
        proxy_server.terminate()
        proxy_server.join()
        sys.exit(1)

    # Keep the main process running and handle signals
    while server_process.is_alive() or proxy_server.is_alive(): 
        server_process.join(timeout=3.0)
        proxy_server.join(timeout=3.0)
            
    # except KeyboardInterrupt:
    #     print("\nReceived keyboard interrupt. Shutting down...")
    # except Exception as e:
    #     print(f"\nUnexpected error: {e}")
    # finally:
    #     # Ensure clean shutdown
    #     if server_process.is_alive():
    #         print("Terminating server process...")
    #         server_process.terminate()
    #         server_process.join(timeout=5.0)
    #         if server_process.is_alive():
    #             print("Force killing server process...")
    #             server_process.kill()
    #             server_process.join()
    #     print("Shutdown complete.")


if __name__ == "__main__":
    cli_env_setup()
    main()
