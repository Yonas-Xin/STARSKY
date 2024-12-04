import setuptools  # 导入setuptools打包工具

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取 requirements.txt 中的依赖
with open("requirements.txt", "r",encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="skystar",
    version="1.0.45",
    author="Yonas-xin",
    author_email="linkstar443@163.com",
    description="A Deep Learning Framework For Beginners",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yonas-Xin/STARSKY",
    packages=['skystar', 'skystar.sky_dataset'],
    install_requires=requirements,  # 从 requirements.txt 读取依赖
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Minimum Python version required

    entry_points={
        'console_scripts': [
            'skystar = skystar.__main__:main'
        ]}
)
