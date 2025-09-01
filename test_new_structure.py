#!/usr/bin/env python3
"""
Test script to verify the new operator structure works correctly
"""

import os
import sys

def test_import_structure():
    """Test that we can import from the new operator structure"""
    print("Testing new operator structure...")
    
    # Add operators to Python path
    operators_path = os.path.join(os.path.dirname(__file__), 'operators')
    if operators_path not in sys.path:
        sys.path.insert(0, operators_path)
    
    try:
        # Test voxelization imports
        print("Testing voxelization operator...")
        from voxelization import voxelize_points, create_voxelization_op, VoxelizationOp
        print("✅ Voxelization imports successful")
        
        # Test NMS imports  
        print("Testing NMS operator...")
        from nms import nms_3d, create_nms_op, NMSOp
        print("✅ NMS imports successful")
        
        # Test BEVPool imports
        print("Testing BEVPool operator...")
        from bevpool import bevpool_forward, create_bevpool_op, BEVPoolOp
        print("✅ BEVPool imports successful")
        
        # Test main operators module
        print("Testing main operators module...")
        import operators
        print(f"Available operators: {operators.__all__}")
        print("✅ Main operators module successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_locations():
    """Test that all required files are in the correct locations"""
    print("\nTesting file locations...")
    
    base_dir = os.path.dirname(__file__)
    required_files = [
        # Voxelization operator
        'operators/voxelization/__init__.py',
        'operators/voxelization/voxelization_kernel.cu',
        'operators/voxelization/voxelization_kernel.h', 
        'operators/voxelization/voxelization_pytorch.py',
        'operators/voxelization/voxelization_binding.cpp',
        
        # NMS operator
        'operators/nms/__init__.py',
        'operators/nms/nms_kernel.cu',
        'operators/nms/nms_kernel.h',
        'operators/nms/nms_pytorch.py', 
        'operators/nms/nms_binding.cpp',
        
        # BEVPool operator
        'operators/bevpool/__init__.py',
        'operators/bevpool/bevpool_kernel.cu',
        'operators/bevpool/bevpool_kernel.h',
        'operators/bevpool/bevpool_pytorch.py',
        'operators/bevpool/bevpool_binding.cpp',
        
        # Main operators module
        'operators/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found")
        return True

def main():
    """Run all tests"""
    print("AI Benchmark - New Structure Verification")
    print("=" * 50)
    
    # Test file locations
    files_ok = test_file_locations()
    
    # Test imports
    imports_ok = test_import_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if files_ok and imports_ok:
        print("🎉 All tests passed! New structure is working correctly.")
        print("\nStructure Summary:")
        print("📁 operators/")
        print("  ├── voxelization/    # Point cloud voxelization")
        print("  │   ├── *.cu, *.h    # CUDA kernel files") 
        print("  │   ├── *.py         # PyTorch wrapper")
        print("  │   └── *.cpp        # C++ bindings")
        print("  ├── nms/             # Non-Maximum Suppression")
        print("  │   ├── *.cu, *.h    # CUDA kernel files")
        print("  │   ├── *.py         # PyTorch wrapper") 
        print("  │   └── *.cpp        # C++ bindings")
        print("  └── bevpool/         # BEV Pooling")
        print("      ├── *.cu, *.h    # CUDA kernel files")
        print("      ├── *.py         # PyTorch wrapper")
        print("      └── *.cpp        # C++ bindings")
        
        print("\nUsage:")
        print("from operators.voxelization import voxelize_points")
        print("from operators.nms import nms_3d") 
        print("from operators.bevpool import bevpool_forward")
        
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)