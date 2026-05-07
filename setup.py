from setuptools import find_packages, setup

setup(
    name="spherope",
    version="0.1.0",
    description="SpheRoPE: zero-shot 360° ERP panorama generation with spherical RoPE, on top of diffusers and LTX-2.3.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "diffusers>=0.37.1",
        # transformers 5.x currently breaks FLUX.2 pipeline loading; keep on 4.x.
        "transformers>=4.45.0,<5.0",
        "accelerate>=0.30.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
    ],
)
