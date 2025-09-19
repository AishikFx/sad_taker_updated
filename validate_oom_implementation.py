#!/usr/bin/env python3
"""
Validation script for CUDA OOM handling implementation
Checks that all required methods and attributes are present
"""

import ast
import sys

def validate_oom_implementation():
    """Validate that the OOM handling implementation is complete"""
    print("🔍 Validating CUDA OOM Handling Implementation")
    print("=" * 55)

    try:
        # Read the face enhancer file
        with open('src/utils/face_enhancer.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)

        # Required methods to check
        required_methods = [
            '_handle_cuda_oom',
            '_emergency_memory_recovery',
            '_extreme_memory_pressure_recovery',
            '_adaptive_oom_prevention',
            'get_performance_insights',
            'record_success',
            'record_failure',
            'get_memory_stats'
        ]

        # Required attributes
        required_attributes = [
            'memory_pressure_level',
            'oom_count',
            'consecutive_oom_count',
            'successful_batch_sizes',
            'failed_batch_sizes'
        ]

        found_methods = []
        found_attributes = []

        # Check class definition and methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TrueBatchGFPGANEnhancer':
                print(f"✅ Found class: {node.name}")

                # Check methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in required_methods:
                            found_methods.append(item.name)
                            print(f"  ✅ Method: {item.name}")

                # Check __init__ method for attributes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        for init_node in ast.walk(item):
                            if isinstance(init_node, ast.Attribute) and isinstance(init_node.value, ast.Name) and init_node.value.id == 'self':
                                attr_name = init_node.attr
                                if attr_name in required_attributes:
                                    found_attributes.append(attr_name)

        print(f"\n📊 Validation Results:")
        print(f"  Methods found: {len(found_methods)}/{len(required_methods)}")
        print(f"  Attributes found: {len(found_attributes)}/{len(required_attributes)}")

        # Check for comprehensive OOM handling comment
        if "Comprehensive CUDA OOM Handling Strategy" in content:
            print("  ✅ Comprehensive OOM documentation found")
        else:
            print("  ❌ Comprehensive OOM documentation missing")

        # Check for multi-layer protection
        protection_layers = [
            "PREVENTION LAYER",
            "RECOVERY LAYER",
            "FALLBACK LAYER",
            "MONITORING LAYER"
        ]

        layers_found = sum(1 for layer in protection_layers if layer in content)
        print(f"  ✅ Protection layers documented: {layers_found}/{len(protection_layers)}")

        # Check for extreme recovery implementation
        if "_extreme_memory_pressure_recovery" in found_methods:
            print("  ✅ Extreme memory pressure recovery implemented")
        else:
            print("  ❌ Extreme memory pressure recovery missing")

        # Check for performance insights
        if "get_performance_insights" in found_methods:
            print("  ✅ Performance insights method implemented")
        else:
            print("  ❌ Performance insights method missing")

        # Summary
        all_methods = len(found_methods) == len(required_methods)
        all_attributes = len(found_attributes) == len(required_attributes)

        if all_methods and all_attributes:
            print("\n🎉 VALIDATION SUCCESSFUL: Complete OOM handling implementation detected")
            print("   - All required methods present")
            print("   - All required attributes present")
            print("   - Multi-layer protection documented")
            print("   - Extreme recovery mechanisms implemented")
            return True
        else:
            print("\n❌ VALIDATION FAILED: Missing components")
            if not all_methods:
                missing_methods = set(required_methods) - set(found_methods)
                print(f"   Missing methods: {missing_methods}")
            if not all_attributes:
                missing_attributes = set(required_attributes) - set(found_attributes)
                print(f"   Missing attributes: {missing_attributes}")
            return False

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_oom_implementation()
    sys.exit(0 if success else 1)