"""
api.py — la "puerta" (API) para que una página HTML pueda usar a Clarita.

Correr:  uvicorn api:app --reload
Queda escuchando en http://127.0.0.1:8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from clasificador import clasificar_texto

app = FastAPI()

# Permite que una página HTML (que vive en otro lado) llame a esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class Entrada(BaseModel):
    texto: str


@app.post("/etiquetar")
def etiquetar(entrada: Entrada):
    """Recibe un texto y devuelve sus etiquetas en formato JSON."""
    etiquetas = clasificar_texto(entrada.texto)
    return {"etiquetas": sorted(etiquetas)}
