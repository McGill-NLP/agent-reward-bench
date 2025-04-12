from setuptools import setup, find_packages

package_name = "agent_reward_bench"

version = {}
with open(f"{package_name}/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

extras_require = {
    "dev": ["black", "wheel"],
    'extra': ['huggingface-hub', 'orjson']
}
# Dynamically create the 'all' extra by combining all other extras
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name=package_name,
    version=version["__version__"],
    author="McGill NLP",
    author_email=f"{package_name}@googlegroups.com",
    url=f"https://github.com/McGill-NLP/{package_name}",
    description=f"The official {package_name} library",
    long_description=long_description,
    packages=find_packages(include=[f"{package_name}*"]),
    package_data={f"{package_name}.data": ["data/*.json", "data/*.csv"]},
    install_requires=[
        "browsergym",
        "numpy",
        "pandas",
        "Pillow",
        "tqdm",
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    # Cast long description to markdown
    long_description_content_type="text/markdown",
)
