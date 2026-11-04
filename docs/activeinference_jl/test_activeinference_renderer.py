#!/usr/bin/env python3
"""
Standalone test script for ActiveInference.jl renderer.
This script tests the renderer without importing the problematic numpy-dependent modules.
"""

import sys
from pathlib import Path

# Add src directory to Python path (from docs/activeinference_jl location)
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_activeinference_renderer():
    """Test the ActiveInference.jl renderer with sample GNN data."""

    # Sample GNN specification (based on the actual JSON export)
    sample_gnn_spec = {
        "name": "Classic Active Inference POMDP Agent v1",
        "description": "Test GNN model for ActiveInference.jl",
        "statespaceblock": [
            {"id": "s", "dimensions": "3,1,type=float"},
            {"id": "o", "dimensions": "3,1,type=int"},
            {"id": "u", "dimensions": "2,type=int"},
            {"id": "A", "dimensions": "3,3,type=float"},
            {"id": "B", "dimensions": "3,3,2,type=float"},
        ],
        "initialparameterization": {
            "A": [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
            "B": [
                [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
                [[0.1, 0.0, 0.9], [0.9, 0.1, 0.0], [0.0, 0.9, 0.1]],
            ],
            "C": [0.0, 0.0, 1.0],
            "D": [0.333, 0.333, 0.334],
            "E": [0.5, 0.5],
        },
    }

    try:
        # Import only the ActiveInference.jl renderer (avoiding numpy)
        from render.activeinference_jl.activeinference_renderer import (
            extract_model_info,
            generate_activeinference_script,
            render_gnn_to_activeinference_jl,
        )

        print("✅ Successfully imported ActiveInference.jl renderer")

        # Test model info extraction
        print("\n🔧 Testing model info extraction...")
        model_info = extract_model_info(sample_gnn_spec)
        print("✅ Extracted model info:")
        print(f"   - Name: {model_info['name']}")
        print(f"   - States: {model_info['n_states']}")
        print(f"   - Observations: {model_info['n_observations']}")
        print(f"   - Actions: {model_info['n_controls']}")
        print(f"   - A matrix: {len(model_info['A'])}x{len(model_info['A'][0])}")
        print(
            f"   - B matrix: {len(model_info['B'])}x{len(model_info['B'][0])}x{len(model_info['B'][0][0])}"
        )

        # Test script generation
        print("\n🔧 Testing script generation...")
        julia_script = generate_activeinference_script(model_info)
        print(f"✅ Generated Julia script ({len(julia_script)} characters)")

        # Test full rendering
        print("\n🔧 Testing full rendering...")
        output_path = Path("test_output") / "test_activeinference.jl"
        output_path.parent.mkdir(exist_ok=True)

        success, message, artifacts = render_gnn_to_activeinference_jl(
            sample_gnn_spec, output_path
        )

        if success:
            print(f"✅ Rendering successful: {message}")
            print(f"   Output file: {artifacts[0]}")

            # Check if file was created
            if output_path.exists():
                print(f"   File size: {output_path.stat().st_size} bytes")

                # Show first few lines
                with open(output_path, "r") as f:
                    lines = f.readlines()[:10]
                    print("   First 10 lines:")
                    for i, line in enumerate(lines, 1):
                        print(f"   {i:2d}: {line.rstrip()}")
            else:
                print("   ❌ Output file not found")
        else:
            print(f"❌ Rendering failed: {message}")

        assert success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing ActiveInference.jl Renderer")
    print("=" * 50)

    success = test_activeinference_renderer()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Tests failed!")

    sys.exit(0 if success else 1)
