from setuptools import setup, find_packages


setup(
    name="optim_sentence_transformrs",
    version="0.1",
    author="sidhantls",
    description="API to optimize SentenceTransformer models using ONNX/Optimum and perform inference using the same `model.encode` API.",
    license="Apache License 2.0",
    packages=find_packages(),
    download_url="https://github.com/sidhantls/optim-sentence-transformers/",
    python_requires=">=3.8.0",
    install_requires=[
        'optimum[onnxruntime]>=1.10.0',
        'sentence-transformers>= 1.0.0',
    ],

    keywords="Onnx Optimized Sentence Transformer BERT sentence embedding"
)