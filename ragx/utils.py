import os
import psutil

# Utility functions
def get_thread_count():
    available_memory = psutil.virtual_memory().available / (1024 ** 2)
    cpu_count = os.cpu_count() or 4
    return max(2, cpu_count // 2) if available_memory < 2000 else cpu_count

# Specification patterns for matching
SPEC_PATTERNS = {
    'ram': r'(\d+)\s*gb\s*ram',
    'storage': r'(\d+)\s*gb\s*(?:rom|storage)',
    'camera': r'(\d+)\s*mp\s*camera',
    'battery': r'(\d+)\s*mah\s*battery',
}
