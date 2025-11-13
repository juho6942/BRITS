"""Quick test to verify NaN fix in beijing_handler."""
import beijing_handler
import numpy as np

print("Testing NaN fix...")
print("-" * 60)

# Load test data
test_loader = beijing_handler.get_loader(batch_size=2, shuffle=False, is_train=False)
print(f"✓ Test loader created with {len(test_loader.dataset)} windows")

# Get one batch
batch = next(iter(test_loader))
print(f"✓ Batch loaded")

# Check for NaN in the data
forward_values = batch['forward']['values'].numpy()
forward_evals = batch['forward']['evals'].numpy()
forward_eval_masks = batch['forward']['eval_masks'].numpy()

print(f"\nChecking for NaN values:")
print(f"  Values shape: {forward_values.shape}")
print(f"  Evals shape: {forward_evals.shape}")
print(f"  Eval masks shape: {forward_eval_masks.shape}")

# Check for NaN in values (should be 0 where masked)
has_nan_values = np.isnan(forward_values).any()
print(f"  Values contain NaN: {has_nan_values}")

# Check for NaN in evals (should NOT have NaN)
has_nan_evals = np.isnan(forward_evals).any()
print(f"  Evals contain NaN: {has_nan_evals}")

if has_nan_evals:
    print(f"  ✗ ERROR: Evals should NOT contain NaN!")
    print(f"  NaN count in evals: {np.isnan(forward_evals).sum()}")
else:
    print(f"  ✓ Evals are clean (no NaN)")

# Check eval_masks
eval_mask_sum = forward_eval_masks.sum()
print(f"\n  Eval masks sum (positions to evaluate): {eval_mask_sum}")

# Check a few ground truth values where eval_mask=1
masked_positions = np.where(forward_eval_masks == 1)
if len(masked_positions[0]) > 0:
    sample_evals = forward_evals[masked_positions][:10]
    print(f"  Sample ground truth values: {sample_evals}")
    print(f"  Min: {sample_evals.min():.4f}, Max: {sample_evals.max():.4f}")

print("\n" + "=" * 60)
if not has_nan_evals:
    print("✓ TEST PASSED - No NaN in ground truth values!")
else:
    print("✗ TEST FAILED - NaN found in ground truth!")
