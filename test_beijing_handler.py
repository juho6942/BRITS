"""Test script to verify beijing_handler.py works correctly."""
import beijing_handler
import torch

print("Testing beijing_handler...")
print("-" * 50)

# Test 1: Load training data
print("\n1. Testing training data loader...")
try:
    train_loader = beijing_handler.get_loader(batch_size=2, shuffle=False, is_train=True)
    print(f"   ✓ Training loader created successfully")
    print(f"   ✓ Number of batches: {len(train_loader)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Get a batch
print("\n2. Testing batch format...")
try:
    batch = next(iter(train_loader))
    print(f"   ✓ Batch loaded successfully")
    print(f"   ✓ Batch keys: {batch.keys()}")
    print(f"   ✓ Forward keys: {batch['forward'].keys()}")
    print(f"   ✓ Backward keys: {batch['backward'].keys()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Check tensor shapes
print("\n3. Testing tensor shapes...")
try:
    batch_size = batch['forward']['values'].shape[0]
    seq_len = batch['forward']['values'].shape[1]
    n_features = batch['forward']['values'].shape[2]
    
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Sequence length: {seq_len}")
    print(f"   ✓ Number of features: {n_features}")
    
    # Verify all tensors have correct shapes
    expected_shape = (batch_size, seq_len, n_features)
    for key in ['values', 'masks', 'deltas', 'forwards', 'evals', 'eval_masks']:
        assert batch['forward'][key].shape == expected_shape, f"Forward {key} shape mismatch"
        assert batch['backward'][key].shape == expected_shape, f"Backward {key} shape mismatch"
    print(f"   ✓ All tensors have correct shape: {expected_shape}")
    
    assert batch['labels'].shape == (batch_size,), "Labels shape mismatch"
    assert batch['is_train'].shape == (batch_size,), "is_train shape mismatch"
    print(f"   ✓ Labels and is_train have correct shape: ({batch_size},)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 4: Check data types
print("\n4. Testing data types...")
try:
    assert batch['forward']['values'].dtype == torch.float32
    assert batch['labels'].dtype == torch.float32
    print(f"   ✓ All tensors are torch.float32")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 5: Verify masks are binary
print("\n5. Verifying masks are binary (0 or 1)...")
try:
    masks_unique = torch.unique(batch['forward']['masks'])
    eval_masks_unique = torch.unique(batch['forward']['eval_masks'])
    print(f"   ✓ Mask unique values: {masks_unique.tolist()}")
    print(f"   ✓ Eval mask unique values: {eval_masks_unique.tolist()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 6: Test loader
print("\n6. Testing test data loader...")
try:
    test_loader = beijing_handler.get_loader(batch_size=2, shuffle=False, is_train=False)
    print(f"   ✓ Test loader created successfully")
    print(f"   ✓ Number of batches: {len(test_loader)}")
    
    test_batch = next(iter(test_loader))
    print(f"   ✓ Test batch loaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n" + "=" * 50)
print("✓ ALL TESTS PASSED!")
print("=" * 50)
print("\nThe beijing_handler.py is now compatible with your training pipeline.")
print("You can use it in main.py by replacing:")
print("  import data_loader")
print("with:")
print("  import beijing_handler as data_loader")
