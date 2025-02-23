from setuptools import setup, find_packages

"""
Argumentos de la función setup():
name: El nombre del paquete que se está creando.
version: La versión del proyecto (útil para actualizaciones).
description: Una breve descripción de lo que hace el proyecto.
author: Mi nombre como creador del paquete.
packages: Detecta automáticamente los submódulos dentro de src (como utils, models, etc.).
install_requires:   Lista de bibliotecas necesarias para el proyecto. 
                    Estas se instalarán automáticamente cuando alguien instale el paquete.
python_requires: Especifica la versión mínima de Python compatible.
"""

setup(
    name="SistemaAI_ACV",                     # Nombre del proyecto
    version="0.1",                            # Versión inicial
    description="Sistema AI Multimodal para diagnóstico de ACV con Faster R-CNN y BERT",  # Descripción
    author="Jorge Leonardo Torres Arévalo",   # Creador del paquete                
    author_email="geoleonardo@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JorgeLeonardoTorres/SistemaAI_ACV",  # URL del repositorio en GitHub
    packages=find_packages(where="src"),      # Buscar paquetes dentro de "src"
    package_dir={"": "src"},                  # Define "src" como directorio base
    include_package_data=True,  # Incluye archivos adicionales (como modelos preentrenados)
    install_requires=[                        # Dependencias necesarias
        "torch",
        "torchvision",
        "numpy",
        "pydicom",
        "tqdm",
        "Pillow",
        "matplotlib"
    ],  # Dependencias
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",                  # Versión mínima de Python
)