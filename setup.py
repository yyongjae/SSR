import os
import re
from pathlib import Path

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

def read_requirements(path: str):
    """Convert pip-style requirements (blank lines/comments included) to PEP 508."""
    requirements = []
    with open(path, encoding="utf-8") as requirement_file:
        for raw_line in requirement_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # A comment starts only after whitespace, so URL fragments would be
            # preserved if a future direct reference needs one.
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if line:
                requirements.append(line)
    return requirements


requirements = read_requirements("requirements_navsim.txt")
config_files = [
    str(path.relative_to("navsim"))
    for path in Path("navsim/planning/script/config").rglob("*.yaml")
]

def main() -> None:
    setuptools.setup(
        name="para-ssr-navsim",
        version="1.0.0",
        author="University of Tuebingen",
        author_email="kashyap.chitta@uni-tuebingen.de",
        description="PARA-SSR port for NAVSIM",
        python_requires=">=3.9",
        packages=setuptools.find_packages(script_folder),
        package_dir={"": "."},
        package_data={"navsim": config_files},
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Operating System :: OS Independent",
            "License :: Free for non-commercial use",
        ],
        license="apache-2.0",
        install_requires=requirements,
    )


if __name__ == "__main__":
    main()
