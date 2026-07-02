"""Página: match de dos niveles (obligatorios vs deseables) entre trabajadores y una oferta."""
import streamlit as st
from estilo import aplicar_estilo
from perfiles import trabajadores, ofertas, match

st.set_page_config(page_title="Match · MineClass", page_icon="🎯", layout="wide")
aplicar_estilo()

st.title("🎯 Match")
st.caption("Elige una oferta y mira quién calza mejor. Los requisitos OBLIGATORIOS "
           "pesan más; los DESEABLES suman puntos extra.")

elegido = st.selectbox("Elige una oferta", [o["cargo"] for o in ofertas])
oferta = next(o for o in ofertas if o["cargo"] == elegido)

st.markdown("**Obligatorios:** " + "; ".join(oferta["excluyentes_texto"] + oferta["excluyentes_perks"]))
deseables = oferta["deseables_texto"] + oferta["deseables_perks"]
st.markdown("**Deseables:** " + ("; ".join(deseables) if deseables else "—"))
st.divider()

resultados = sorted(
    [(t["nombre"], match(t, oferta)) for t in trabajadores],
    key=lambda x: x[1]["porcentaje"], reverse=True,
)
for nombre, r in resultados:
    estado = "✅ cumple obligatorios" if r["cumple_obligatorios"] else "❌ le faltan obligatorios"
    st.markdown(f"**{nombre}** — {r['porcentaje']}%  ·  {estado}")
    st.progress(r["porcentaje"] / 100)
    detalle = (f"Obligatorios: {r['obligatorios'][0]}/{r['obligatorios'][1]}  ·  "
               f"Deseables: {r['deseables'][0]}/{r['deseables'][1]}")
    if r["falta_obligatorio"]:
        detalle += "  ·  Le falta: " + ", ".join(r["falta_obligatorio"])
    st.caption(detalle)
