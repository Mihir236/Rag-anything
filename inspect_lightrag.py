import lightrag
import sys
import pprint

try:
    from lightrag.kg import shared_storage
    print("\n--- lightrag.kg.shared_storage dir ---")
    pprint.pprint(dir(shared_storage))
    
    # Check for anything generic like 'storages', 'registry', 'cache'
    for attr in dir(shared_storage):
        val = getattr(shared_storage, attr)
        if isinstance(val, dict):
             print(f"\nPotential Dict Cache: {attr} = {val.keys()}")
except ImportError:
    print("Could not import lightrag.kg.shared_storage")

try:
    import lightrag.utils
    print("\n--- lightrag.utils dir ---")
    pprint.pprint(dir(lightrag.utils))
except ImportError:
    pass
