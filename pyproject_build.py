"""
this script is used to dynamically generate the version number for the package during build
"""

def get_version():
    with open("version.md", "r") as f:
        return f.read().strip()

def build_meta():
    return {
        "version": get_version(),
    }