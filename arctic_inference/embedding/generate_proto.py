#!/usr/bin/env python3
"""
Standalone script to generate gRPC code from the proto file.
This script is designed to be run directly without importing any modules
that might cause circular imports.
"""

import os
import subprocess
import sys
from pathlib import Path


def generate_grpc_code():
    """Generate gRPC Python code from proto file."""
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    proto_dir = current_dir / "proto"
    proto_file = proto_dir / "inference.proto"

    # Ensure the proto file exists
    if not proto_file.exists():
        print(f"Error: Proto file not found: {proto_file}")
        return 1

    # Create the command to generate Python code
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={proto_dir}/python/",
        f"--grpc_python_out={proto_dir}/python/",
        str(proto_file),
    ]

    if not os.path.exists(proto_dir / "python"):
        os.makedirs(proto_dir / "python")

    # Run the command
    print(f"Generating gRPC code from {proto_file}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Successfully generated gRPC code")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate gRPC code: {e.stderr}")
        return 1

    # Fix imports in the generated files
    pb2_file = proto_dir / "python/inference_pb2.py"
    pb2_grpc_file = proto_dir / "python/inference_pb2_grpc.py"

    if pb2_file.exists() and pb2_grpc_file.exists():
        # Fix imports in the grpc file
        with open(pb2_grpc_file, "r") as f:
            content = f.read()

        # Replace relative import with absolute import
        content = content.replace(
            "import inference_pb2 as inference__pb2",
            "from arctic_inference.embedding.proto.python import inference_pb2 as inference__pb2",
        )

        with open(pb2_grpc_file, "w") as f:
            f.write(content)

        print("Fixed imports in generated files")

        print("\nGRPC code generation complete!")
        print("Generated files:")
        print(f"- {pb2_file}")
        print(f"- {pb2_grpc_file}")
        return 0
    else:
        print("Error: Generated files not found")
        return 1

if __name__ == "__main__":
    sys.exit(generate_grpc_code())
