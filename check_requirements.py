# check_requirements.py
print("🚀 Le script se lance bien")

import chardet

def read_normalized(path):
    # Detect encoding
    with open(path, 'rb') as f:
        raw = f.read()
    encoding = chardet.detect(raw)['encoding']
    
    # Decode with detected encoding
    text = raw.decode(encoding)
    
    # Normalize package names (lowercase, replace _ by -)
    return {
        line.strip().split("==")[0].lower().replace("_", "-")
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }

installed = read_normalized("installed.txt")
requirements = read_normalized("requirements.txt")

missing_in_requirements = installed - requirements
missing_installed = requirements - installed

print("Packages installés mais absents du requirements :", missing_in_requirements)
print("Packages listés dans requirements mais pas installés :", missing_installed)
