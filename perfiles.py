"""
perfiles.py
===========
Perfiles de trabajadores + ofertas de trabajo (con requisitos EXCLUYENTES y
DESEABLES), analizados por Clarita, y el MATCH de dos niveles entre ellos.

Corre con:  python perfiles.py   (necesita clasificador.py en la misma carpeta)
"""

from clasificador import clasificar_texto


# ----------------------------------------------------------------------
# PERKS: lista fija de la que se ELIGE (como casillas). No necesitan IA.
# ----------------------------------------------------------------------
PERKS_POSIBLES = [
    "licencia_conducir", "vehiculo_propio", "disponible_turnos",
    "disponibilidad_inmediata", "disponible_viajar", "ingles", "todo_tipo_contrato",
]


# ----------------------------------------------------------------------
# PERFILES DE TRABAJADORES (de ejemplo; reemplázalos por los tuyos)
# ----------------------------------------------------------------------
trabajadores = [
    {
        "nombre": "Carlos Medina",
        "sobre_mi": "Operador de equipos pesados con 6 años en faena minera, enfocado en la seguridad.",
        "trayectoria": [
            "Operé camiones mineros y cargador frontal en mina a rajo abierto",
            "Mantención mecánica básica de la maquinaria a mi cargo",
        ],
        "perks": ["licencia_conducir", "disponible_turnos", "disponibilidad_inmediata"],
    },
    {
        "nombre": "Andrea Varas",
        "sobre_mi": "Ingeniera en control de gestión con experiencia en recursos humanos y finanzas.",
        "trayectoria": [
            "Jefe de administración y recursos humanos zona norte",
            "Reclutamiento, selección y contratación de personal",
        ],
        "perks": ["ingles", "disponible_viajar", "licencia_conducir"],
    },
    {
        "nombre": "Favio Barahona",
        "sobre_mi": "Ingeniero planificador de proyectos con experiencia en minería.",
        "trayectoria": [
            "Planificador de proyectos en empresa minera con manejo de cronogramas",
            "Mantenedor mecánico de equipos y maquinaria",
        ],
        "perks": ["ingles", "disponibilidad_inmediata"],
    },
    {
        "nombre": "José Fuentes",
        "sobre_mi": "Soldador calificado con experiencia en maestranza y estructuras metálicas.",
        "trayectoria": [
            "Soldadura al arco, MIG y TIG de estructuras y cañerías",
            "Reparación mecánica de componentes en equipos",
        ],
        "perks": ["disponible_turnos", "disponibilidad_inmediata", "todo_tipo_contrato"],
    },
    {
        "nombre": "Camila Rojas",
        "sobre_mi": "Prevencionista de riesgos con experiencia en faena y apoyo administrativo.",
        "trayectoria": [
            "Experta en prevención de riesgos y seguridad ocupacional en faena",
            "Inspecciones de seguridad y charlas de prevención de riesgos",
        ],
        "perks": ["licencia_conducir", "disponible_turnos", "ingles"],
    },
]


# ----------------------------------------------------------------------
# OFERTAS DE TRABAJO (con dos niveles de requisitos)
#   excluyentes_texto / excluyentes_perks -> OBLIGATORIOS (sin esto, no calza)
#   deseables_texto   / deseables_perks   -> IDEALES (suman puntos, no obligan)
# ----------------------------------------------------------------------
ofertas = [
    {
        "cargo": "Operador de equipos pesados",
        "excluyentes_texto": ["Operar camiones mineros y maquinaria pesada"],
        "excluyentes_perks": ["licencia_conducir", "disponible_turnos"],
        "deseables_texto": ["Conocimientos de mantención mecánica"],
        "deseables_perks": ["disponibilidad_inmediata"],
    },
    {
        "cargo": "Jefe de Recursos Humanos",
        "excluyentes_texto": [
            "Gestión de recursos humanos y reclutamiento de personal",
            "Administración de personal y remuneraciones",
        ],
        "excluyentes_perks": [],
        "deseables_texto": ["Capacitación y desarrollo del personal"],
        "deseables_perks": ["ingles"],
    },
    {
        "cargo": "Prevencionista de Riesgos",
        "excluyentes_texto": ["Prevención de riesgos y seguridad ocupacional en faena"],
        "excluyentes_perks": ["licencia_conducir", "disponible_turnos"],
        "deseables_texto": ["Inspecciones de seguridad y uso de EPP"],
        "deseables_perks": [],
    },
    {
        "cargo": "Soldador Estructural",
        "excluyentes_texto": ["Soldadura al arco de estructuras y cañerías"],
        "excluyentes_perks": ["disponible_turnos"],
        "deseables_texto": ["Mantención mecánica de equipos"],
        "deseables_perks": ["todo_tipo_contrato"],
    },
]


# ----------------------------------------------------------------------
# ANALIZAR: convertir textos en un CONJUNTO de etiquetas (usando Clarita)
# ----------------------------------------------------------------------
def etiquetas_de_textos(lista_textos):
    todas = set()
    for texto in lista_textos:
        todas |= clasificar_texto(texto)
    return todas


def analizar_trabajador(t):
    textos = [t["sobre_mi"]] + t["trayectoria"]
    return {"skills": etiquetas_de_textos(textos), "perks": set(t["perks"])}


def analizar_oferta(o):
    # obligatorios (must) y deseables (nice), cada uno = etiquetas de texto + perks
    must = etiquetas_de_textos(o["excluyentes_texto"]) | set(o["excluyentes_perks"])
    nice = etiquetas_de_textos(o["deseables_texto"]) | set(o["deseables_perks"])
    return {"must": must, "nice": nice}


# ----------------------------------------------------------------------
# MATCH de dos niveles.
#   - Cumplir OBLIGATORIOS vale el 70% del puntaje.
#   - Cumplir DESEABLES vale el 30% (bonus).
#   - Si le falta algún obligatorio, se marca la advertencia.
# ----------------------------------------------------------------------
def match(trabajador, oferta):
    tiene = analizar_trabajador(trabajador)
    tiene_todo = tiene["skills"] | tiene["perks"]

    o = analizar_oferta(oferta)
    must, nice = o["must"], o["nice"]

    must_ok = must & tiene_todo
    must_falta = must - tiene_todo
    nice_ok = nice & tiene_todo

    ratio_must = len(must_ok) / len(must) if must else 1.0
    ratio_nice = len(nice_ok) / len(nice) if nice else 0.0

    if nice:
        porcentaje = round(100 * (0.7 * ratio_must + 0.3 * ratio_nice))
    else:
        porcentaje = round(100 * ratio_must)

    return {
        "porcentaje": porcentaje,
        "cumple_obligatorios": len(must_falta) == 0,
        "obligatorios": (len(must_ok), len(must)),   # (cumplidos, total)
        "deseables": (len(nice_ok), len(nice)),
        "falta_obligatorio": sorted(must_falta),
        "tiene_deseables": sorted(nice_ok),
    }


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for oferta in ofertas:
        print("=" * 60)
        print(f"OFERTA: {oferta['cargo']}")
        print("=" * 60)

        resultados = sorted(
            [(t["nombre"], match(t, oferta)) for t in trabajadores],
            key=lambda x: x[1]["porcentaje"], reverse=True,
        )
        for nombre, r in resultados:
            ok = "✅ cumple obligatorios" if r["cumple_obligatorios"] else "❌ le faltan obligatorios"
            print(f"\n  {nombre}: {r['porcentaje']}%   {ok}")
            print(f"     obligatorios: {r['obligatorios'][0]}/{r['obligatorios'][1]}   "
                  f"deseables: {r['deseables'][0]}/{r['deseables'][1]}")
            if r["falta_obligatorio"]:
                print(f"     le falta (obligatorio): {', '.join(r['falta_obligatorio'])}")
        print()
