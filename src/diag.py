"""Quick diagnostic — run this to check what's available in mediapipe."""
import mediapipe as mp

print(f"mediapipe version : {mp.__version__}")
print(f"mp.solutions exists: {hasattr(mp, 'solutions')}")
print()

# Check tasks API
try:
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    print("mediapipe.tasks API : AVAILABLE")
except Exception as e:
    print(f"mediapipe.tasks API : NOT available ({e})")

# Check solutions API
try:
    import mediapipe.python.solutions.hands as _hands
    print("solutions.hands     : AVAILABLE")
except Exception as e:
    print(f"solutions.hands     : NOT available ({e})")
