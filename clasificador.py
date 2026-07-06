"""
clasificador.py
===============
Tu clasificador entrenado con ejemplos (machine learning de verdad).

NUEVO en esta versión:
1) Los ejemplos están ORDENADOS por categoría, para que encuentres y agregues fácil.
2) Hay una función normalizar() que pasa el texto a MINÚSCULAS y le saca las TILDES.
   IMPORTANTE: se la aplicamos a los ejemplos de entrenamiento Y a los textos nuevos.
   (más abajo te explico por qué tiene que ser a los dos)

Para correrlo:
    pip install scikit-learn
    python clasificador.py
"""

import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


# ======================================================================
# 0) NORMALIZAR  (limpiar el texto antes de usarlo)
#    - lo pasa a minúsculas
#    - le quita las tildes (á -> a, é -> e, etc.)
#
#    ¿Por qué hay que aplicarla a los DOS lados (ejemplos y textos nuevos)?
#    Porque el modelo compara PALABRAS. Si entrena con "Operé" pero le llega
#    "opere", para él son dos palabras distintas y no calzan. Al limpiar ambos
#    igual, "Operé" y "opere" se vuelven "opere" en los dos lados y SÍ calzan.
#    (Ojo: esto también convierte la ñ en n, pero como pasa en ambos lados,
#     no causa problemas.)
# ======================================================================
def normalizar(texto):
    texto = texto.lower()                          # a minúsculas
    texto = unicodedata.normalize("NFD", texto)    # separa la letra de su tilde
    texto = "".join(c for c in texto               # bota solo las tildes
                    if unicodedata.category(c) != "Mn")
    return texto


# ======================================================================
# 1) LOS EJEMPLOS  (de aquí APRENDE el modelo)
#    Ordenados por categoría. Para agregar uno nuevo, búscalo por su grupo.
#    Regla de oro de las etiquetas: minúsculas, sin tildes y con guion_bajo.
# ======================================================================
ejemplos = [

    # ---------- OPERADOR DE MAQUINARIA PESADA ----------
    ("Operé camiones mineros y cargador frontal en faena por 5 años", ["operador_maquinaria_pesada"]),
    ("Conducción de maquinaria pesada de extracción en mina a rajo abierto", ["operador_maquinaria_pesada"]),
    ("Manejo de equipos pesados y palas en terreno", ["operador_maquinaria_pesada"]),
    ("Operador de perforadora y camión de alto tonelaje", ["operador_maquinaria_pesada"]),
    ("Licencia tipo D", ["operador_maquinaria_pesada"]),
    ("Experiencia manejando con maquinaria pesada", ["operador_maquinaria_pesada"]),
    ("Conductor de camión tolva de alto tonelaje en mina", ["operador_maquinaria_pesada"]),
    ("Operador de pala hidráulica y excavadora", ["operador_maquinaria_pesada"]),
    ("Manejo de bulldozer y motoniveladora en terreno", ["operador_maquinaria_pesada"]),
    ("Operador de retroexcavadora y camión aljibe", ["operador_maquinaria_pesada"]),
    ("Operador de maquinaria pesada en faena minera", ["operador_maquinaria_pesada"]),

    # ---------- SOLDADURA ----------
    ("Soldador calificado en soldadura al arco y estructuras metálicas", ["soldadura"]),
    ("Realicé uniones soldadas y reparé estructuras de acero", ["soldadura"]),
    ("Trabajos de soldadura MIG y TIG en maestranza", ["soldadura"]),
    ("Armado y soldadura de cañerías y piezas metálicas", ["soldadura"]),
    ("Soldé uniones", ["soldadura"]),
    # ejemplo con DOS etiquetas, para mostrar que puede elegir varias:
    ("Soldador con experiencia en mantención mecánica de equipos", ["soldadura", "mantencion_maquinaria"]),

    # ---------- MANTENCIÓN DE MAQUINARIA ----------
    ("Mantención mecánica de equipos y reparación de motores diésel", ["mantencion_maquinaria"]),
    ("Mecánico industrial, mantenimiento preventivo de maquinaria", ["mantencion_maquinaria"]),
    ("Reparación mecánica y cambio de componentes en equipos mineros", ["mantencion_maquinaria"]),
    ("Diagnóstico y mantención de sistemas hidráulicos", ["mantencion_maquinaria"]),
    ("Reparación de distintas maquinarias", ["mantencion_maquinaria"]),
    ("Mantenedor mecánico de equipos y maquinaria", ["mantencion_maquinaria"]),
    ("Reparación y cambio de repuestos de motores", ["mantencion_maquinaria"]),
    ("Mantenimiento preventivo y correctivo de maquinaria", ["mantencion_maquinaria"]),
    ("Diagnóstico de fallas mecánicas y reemplazo de piezas desgastadas", ["mantencion_maquinaria"]),
    ("Lubricación, ajuste y overhaul de componentes", ["mantencion_maquinaria"]),
    ("Mantención de maquinaria pesada en taller", ["mantencion_maquinaria"]),

    # ---------- PREVENCIÓN / SEGURIDAD ----------
    ("Experto en prevención de riesgos y seguridad ocupacional en faena", ["prevencion"]),
    ("Encargado de seguridad, protocolos HSEC y uso de EPP", ["prevencion"]),
    ("Supervisión de seguridad y prevención de accidentes en terreno", ["prevencion"]),
    ("Inspecciones de seguridad y charlas de prevención de riesgos", ["prevencion"]),
    ("Enfocado en la seguridad", ["prevencion"]),

    # ---------- ELECTRICIDAD ----------
    ("Mantención eléctrica e instalación de tableros en planta", ["electricidad"]),
    ("Electricista industrial, conexiones y reparación eléctrica", ["electricidad"]),
    ("Diagnóstico de fallas eléctricas y cableado de equipos", ["electricidad"]),
    ("Instalación y mantención de sistemas eléctricos de potencia", ["electricidad"]),

    # ---------- ADMINISTRACIÓN ----------
    ("Ingresar a todo el personal en plataforma de Codelco para su acreditación", ["administracion"]),
    ("Acreditación de trabajadores y carga de documentos en plataforma", ["administracion"]),
    ("Gestión de documentación, contratos y archivo del personal", ["administracion"]),
    ("Registro y actualización de datos de empleados en el sistema", ["administracion"]),
    ("Coordinación administrativa de ingresos, finiquitos y planillas", ["administracion"]),
    ("Tramitación de credenciales y permisos de acceso a faena", ["administracion"]),
    ("Manejo de planillas, correos y agenda de la oficina", ["administracion"]),
    ("Apoyo administrativo en trámites internos y atención de proveedores", ["administracion"]),
    ("Control de asistencia y gestión de turnos del personal", ["administracion"]),
    ("Digitación de información y emisión de reportes administrativos", ["administracion"]),
    ("Administrador de contrato en empresa de servicios", ["administracion_contratos"]),

    # ---------- RRHH / CAPACITACIÓN ----------
    ("Jefe de capacitación y relaciones laborales, gestión de programas formativos", ["jefatura_rrhh", "capacitacion_desarrollo"]),
    ("Asistente y encargada de recursos humanos", ["jefatura_rrhh"]),
    ("Ingeniero en administración de personas y recursos humanos", ["jefatura_rrhh"]),
    ("Encargado de recursos humanos y gestión de personas", ["jefatura_rrhh"]),
    ("Reclutamiento, selección y contratación de personal", ["jefatura_rrhh"]),
    ("Gestión de remuneraciones, nómina y beneficios del personal", ["jefatura_rrhh"]),
    ("Responsable del clima laboral y bienestar de los trabajadores", ["jefatura_rrhh"]),
    ("Evaluación de desempeño y desarrollo de talento interno", ["jefatura_rrhh"]),
    ("Procesos de onboarding e inducción de nuevos colaboradores", ["jefatura_rrhh"]),
    ("Analista de recursos humanos en empresa minera", ["jefatura_rrhh"]),
    ("Generalista de RRHH a cargo de selección y bienestar", ["jefatura_rrhh"]),
    ("Administración de personal y relaciones laborales", ["jefatura_rrhh", "administracion"]),
    ("Gestión de personas, contratación y desvinculaciones", ["jefatura_rrhh", "administracion"]),
    ("Capacitación y formación del personal de la empresa", ["jefatura_rrhh", "capacitacion_desarrollo"]),

    # ---------- JEFATURA / LIDERAZGO / SUPERVISIÓN ----------
    ("Jefe de sucursal, implementación de políticas y supervisión del equipo", ["jefatura_gerencia"]),
    ("Lideré equipos de trabajo de más de 70 personas en distintas áreas", ["liderazgo_equipos"]),
    ("A cargo de equipos de las áreas de mantenimiento, prevención y servicio", ["liderazgo_equipos"]),
    ("Coordiné personal de seguridad, operaciones y mantención en faena", ["liderazgo_equipos"]),
    ("Supervisé jefes de área de distintos departamentos", ["liderazgo_equipos", "supervision_personal"]),
    ("Lideré un equipo de ventas", ["liderazgo_equipos"]),
    ("Liderar equipo de trabajo según estándares y procesos", ["jefatura_gerencia", "supervision_personal"]),

    # ---------- COMERCIAL / VENTAS / RETAIL / HOTELERÍA ----------
    ("Supervisor de supermercado en retail, a cargo del personal de tienda", ["supervision_personal", "retail"]),
    ("Sub-gerente de hotel a cargo de operatividad, turnos y proyección comercial", ["jefatura_gerencia", "hoteleria"]),
    ("Administrador comercial, emisión de facturas y supervisión de personal", ["area_comercial_ventas", "supervision_personal"]),
    ("Sub-gerente comercial en centro comercial, negociación y liderazgo de equipos", ["jefatura_gerencia", "area_comercial_ventas", "retail"]),
    ("Gerente de ventas liderando equipo comercial y control de inventario", ["area_comercial_ventas", "jefatura_gerencia"]),

    # ---------- CONTROL DE GESTIÓN / FINANZAS / COBRANZA / CONSULTORÍA ----------
    ("Titulado en administración de empresas y técnico en comercio exterior", ["administracion_empresas", "comercio_exterior"]),
    ("Ingeniero en control de gestión asesorando pymes en procesos y procedimientos", ["control_gestion", "asesoria_consultoria"]),
    ("Jefe de administración y cobranza, control de procesos contables", ["administracion"]),
    ("Jefe de administración y RRHH zona norte, finanzas y selección de personal", ["jefatura_rrhh", "administracion"]),
    ("Jefe administrativo de sucursal, gestión de recursos y servicios", ["jefatura_gerencia", "administracion"]),
    ("Analista contable y asistente de administración", ["administracion"]),
    ("Titulada en ingeniería de ejecución en control de gestión", ["control_gestion"]),
    ("Asesorías estratégicas y consultoría para el crecimiento de negocios", ["asesoria_consultoria"]),

    # ---------- PLANIFICACIÓN DE PROYECTOS / INGENIERÍA ----------
    ("Ingeniero planificador de proyectos con más de 5 años de experiencia", ["planificacion_proyectos"]),
    ("Planificador de proyectos en empresa minera", ["planificacion_proyectos"]),
    ("Planificador a cargo de cronogramas y control de proyectos", ["planificacion_proyectos"]),
    ("Magíster en dirección y gestión de proyectos", ["planificacion_proyectos"]),
    ("Ingeniero en automatización y control industrial", ["ingenieria"]),

    # ---------- GEOLOGÍA ----------
    ("Geólogo de terreno a cargo de mapeo y muestreo", ["geologia"]),
    ("Logueo de sondajes y descripción de testigos de perforación", ["geologia"]),
    ("Geólogo de exploración minera y modelamiento de yacimientos", ["geologia"]),
    ("Muestreo geológico y control de leyes en mina", ["geologia"]),

    # ---------- TOPOGRAFÍA ----------
    ("Topógrafo de mina con manejo de estación total", ["topografia"]),
    ("Levantamiento topográfico y replanteo en faena", ["topografia"]),
    ("Control topográfico de avances y cálculo de volúmenes", ["topografia"]),
    ("Manejo de GPS y nivel para topografía minera", ["topografia"]),

    # ---------- PERFORACIÓN Y TRONADURA ----------
    ("Perforista de pozos de tronadura en mina a rajo abierto", ["perforacion_tronadura"]),
    ("Manejo de explosivos y carguío de tiros", ["perforacion_tronadura"]),
    ("Operador de equipo de perforación para voladura", ["perforacion_tronadura"]),
    ("Tronadura y voladura controlada en faena", ["perforacion_tronadura"]),

    # ---------- PROCESAMIENTO DE MINERALES / PLANTA ----------
    ("Operador de planta concentradora de minerales", ["procesamiento_minerales"]),
    ("Operación de chancado, molienda y flotación", ["procesamiento_minerales"]),
    ("Control de procesos de lixiviación y SX-EW", ["procesamiento_minerales"]),
    ("Operador de planta de procesamiento en faena", ["procesamiento_minerales"]),

    # ---------- METALURGIA ----------
    ("Ingeniero metalurgista en procesos de recuperación de cobre", ["metalurgia"]),
    ("Optimización de procesos metalúrgicos de la planta", ["metalurgia"]),
    ("Metalurgista a cargo del balance metalúrgico", ["metalurgia"]),
    ("Pruebas metalúrgicas y análisis de recuperación", ["metalurgia"]),

    # ---------- MEDIO AMBIENTE ----------
    ("Encargado de medio ambiente y monitoreo ambiental", ["medio_ambiente"]),
    ("Gestión de permisos ambientales y cumplimiento de la RCA", ["medio_ambiente"]),
    ("Monitoreo de calidad del aire y manejo de residuos", ["medio_ambiente"]),
    ("Profesional de medio ambiente para faena minera", ["medio_ambiente"]),

    # ---------- INSTRUMENTACIÓN Y CONTROL ----------
    ("Instrumentista de control y calibración de sensores", ["instrumentacion_control"]),
    ("Mantención de PLC y sistemas SCADA en planta", ["instrumentacion_control"]),
    ("Control automático de procesos e instrumentación industrial", ["instrumentacion_control"]),
    ("Técnico en instrumentación de procesos mineros", ["instrumentacion_control"]),

    # ---------- LABORATORIO ----------
    ("Analista de laboratorio químico de minerales", ["laboratorio"]),
    ("Ensayos y análisis de muestras en laboratorio", ["laboratorio"]),
    ("Preparación de muestras y química analítica", ["laboratorio"]),
    ("Técnico de laboratorio de ensayes minerales", ["laboratorio"]),

    # ---------- LOGÍSTICA Y BODEGA ----------
    ("Encargado de bodega y control de inventario en faena", ["logistica_bodega"]),
    ("Logística y abastecimiento de insumos mineros", ["logistica_bodega"]),
    ("Gestión de bodega, despacho y recepción de materiales", ["logistica_bodega"]),
    ("Coordinación logística de transporte y suministros", ["logistica_bodega"]),

    # ---------- SALUD OCUPACIONAL / RESCATE ----------
    ("Paramédico de faena y atención de urgencias", ["salud_ocupacional"]),
    ("Enfermería ocupacional en campamento minero", ["salud_ocupacional"]),
    ("Control de exámenes preocupacionales y salud ocupacional", ["salud_ocupacional"]),
    ("Rescate minero y primeros auxilios en terreno", ["salud_ocupacional"]),

    # ---------- CALIDAD (QA/QC) ----------
    ("Inspector de calidad QA QC en proyectos de construcción", ["calidad"]),
    ("Control de calidad de procesos y documentación QC", ["calidad"]),
    ("Inspecciones de calidad y cumplimiento de estándares", ["calidad"]),
    ("Aseguramiento y control de calidad de soldaduras", ["calidad", "soldadura"]),

    # ---------- OPERADOR DE GRÚA / MANIOBRAS ----------
    ("Operador de grúa horquilla y grúa pluma", ["operador_grua"]),
    ("Maniobras de izaje y rigging en faena", ["operador_grua"]),
    ("Rigger señalero a cargo de maniobras de levante", ["operador_grua"]),
    ("Operación de grúa para montaje de estructuras", ["operador_grua"]),

    # ---------- ALIMENTACIÓN / CASINO ----------
    ("Manipulador de alimentos en casino de faena", ["alimentacion_casino"]),
    ("Servicios de alimentación y cocina en campamento minero", ["alimentacion_casino"]),
    ("Garzón y atención de comedor en faena", ["alimentacion_casino"]),

    # ---------- ASEO / SERVICIOS GENERALES ----------
    ("Servicios generales y aseo industrial en faena", ["aseo_servicios"]),
    ("Auxiliar de aseo y limpieza de campamento", ["aseo_servicios"]),
    ("Mantención de áreas comunes y servicios generales", ["aseo_servicios"]),

    # ==================================================================
    # REFUERZO: más ejemplos y vocabulario propio para las categorías flacas
    # ==================================================================

    # --- geología (reforzada, vocabulario propio) ---
    ("Graduado de geología con experiencia en reconocimiento de suelos y minerales", ["geologia"]),
    ("Geólogo dedicado al reconocimiento de minerales y tipos de roca", ["geologia"]),
    ("Reconocimiento de suelos, rocas y minerales en terreno", ["geologia"]),
    ("Cartografía geológica y estimación de recursos de un yacimiento", ["geologia"]),
    ("Geólogo junior en campañas de exploración y sondajes", ["geologia"]),

    # --- laboratorio (afinada: trabajo de laboratorio, NO solo 'minerales') ---
    ("Ensayos de laboratorio por vía húmeda y absorción atómica", ["laboratorio"]),
    ("Preparación de muestras, chancado fino y pulverizado en laboratorio", ["laboratorio"]),
    ("Titulación y análisis químico cuantitativo en laboratorio", ["laboratorio"]),
    ("Laborante a cargo de granulometría y control de calidad de ensayos", ["laboratorio"]),

    # --- topografía ---
    ("Replanteo y monitoreo de deformaciones con estación total", ["topografia"]),
    ("Topógrafo de obra en proyectos de construcción", ["topografia"]),

    # --- perforación y tronadura ---
    ("Ayudante de perforación y carguío de explosivos en banco", ["perforacion_tronadura"]),
    ("Manejo de ANFO y detonadores para voladura", ["perforacion_tronadura"]),

    # --- procesamiento de minerales / planta ---
    ("Operador de espesadores y celdas de flotación", ["procesamiento_minerales"]),
    ("Control de planta de chancado y correas transportadoras", ["procesamiento_minerales"]),

    # --- metalurgia ---
    ("Ingeniero de procesos en planta de lixiviación y electroobtención", ["metalurgia"]),
    ("Control de recuperación metalúrgica y ley de concentrado", ["metalurgia"]),

    # --- medio ambiente ---
    ("Manejo de residuos peligrosos y planes de mitigación ambiental", ["medio_ambiente"]),
    ("Monitoreo de aguas, suelos y calidad del aire en faena", ["medio_ambiente"]),

    # --- instrumentación y control ---
    ("Calibración de transmisores de presión, flujo y temperatura", ["instrumentacion_control"]),
    ("Programación de PLC y lazos de control de procesos", ["instrumentacion_control"]),

    # --- logística y bodega ---
    ("Control de stock, kardex y recepción de mercadería en bodega", ["logistica_bodega"]),
    ("Despacho de insumos y coordinación de camiones de suministro", ["logistica_bodega"]),

    # --- salud ocupacional / rescate ---
    ("Técnico paramédico en policlínico de faena", ["salud_ocupacional"]),
    ("Brigadista de rescate y control de emergencias en mina", ["salud_ocupacional"]),

    # --- calidad (QA/QC) ---
    ("Inspector QA QC de terreno y liberación de protocolos", ["calidad"]),
    ("Control dimensional y trazabilidad de materiales", ["calidad"]),

    # --- operador de grúa / maniobras ---
    ("Operador de grúa telescópica en montaje industrial", ["operador_grua"]),
    ("Señalero y rigger certificado para izajes críticos", ["operador_grua"]),

    # --- alimentación / casino ---
    ("Cocinero de casino en campamento minero", ["alimentacion_casino"]),
    ("Manipulación de alimentos y aseo de cocina en faena", ["alimentacion_casino"]),

    # --- aseo / servicios generales ---
    ("Nochero y servicios de aseo en dependencias de faena", ["aseo_servicios"]),

    # --- administrativas / negocio (refuerzo) ---
    ("Administración de contratos y control de subcontratos", ["administracion_contratos"]),
    ("Facilitador de capacitaciones y desarrollo de competencias", ["capacitacion_desarrollo"]),
    ("Analista de control de gestión e indicadores de desempeño", ["control_gestion"]),
    ("Consultor de negocios y asesoría estratégica a empresas", ["asesoria_consultoria"]),
    ("Ejecutivo de cobranza y recuperación de cartera morosa", ["cobranza"]),
    ("Jefe de tienda en retail y reposición de mercadería", ["retail"]),
    ("Recepcionista y atención al huésped en hotel", ["hoteleria"]),
    ("Ejecutivo de ventas y prospección de clientes", ["area_comercial_ventas"]),
    ("Encargado de importaciones y comercio exterior", ["comercio_exterior"]),
    ("Coordinador de planificación y programación de proyectos", ["planificacion_proyectos"]),
    ("Ingeniero civil industrial en optimización de procesos", ["ingenieria"]),
    ("Supervisor a cargo de la asistencia y desempeño del personal", ["supervision_personal"]),

    # --- refuerzo casos borde ---
    ("Subgerente a cargo de operaciones y de la sucursal", ["jefatura_gerencia"]),
    ("Gerente de operaciones de la empresa", ["jefatura_gerencia"]),
    ("Jefatura general y dirección de la sucursal", ["jefatura_gerencia"]),
    ("Control de la calidad del aire y emisiones atmosféricas", ["medio_ambiente"]),
    ("Gestión ambiental y monitoreo de la calidad del aire", ["medio_ambiente"]),

    # ==================================================================
    # OFERTAS: mismos rubros, pero con TONO DE OFERTA de trabajo
    # ("se requiere", "buscamos", "indispensable", "deseable", "excluyente")
    # ==================================================================
    ("Se requiere operador de camión de extracción y maquinaria pesada para faena", ["operador_maquinaria_pesada"]),
    ("Buscamos mantenedor mecánico para reparación de equipos, excluyente experiencia en motores", ["mantencion_maquinaria"]),
    ("Se necesita soldador calificado para estructuras metálicas y cañerías", ["soldadura"]),
    ("Requerimos electricista industrial para mantención de tableros y equipos", ["electricidad"]),
    ("Se solicita experto en prevención de riesgos para faena, indispensable manejo de seguridad", ["prevencion"]),
    ("Buscamos geólogo para reconocimiento de minerales y logueo de sondajes", ["geologia"]),
    ("Se requiere topógrafo con manejo de estación total para levantamientos", ["topografia"]),
    ("Vacante para perforista de tronadura, deseable manejo de explosivos", ["perforacion_tronadura"]),
    ("Se necesita operador de planta para procesos de flotación y molienda", ["procesamiento_minerales"]),
    ("Requerimos ingeniero metalurgista para procesos de recuperación de cobre", ["metalurgia"]),
    ("Se busca encargado de medio ambiente para monitoreo ambiental y permisos", ["medio_ambiente"]),
    ("Vacante de instrumentista para calibración de sensores y control de procesos", ["instrumentacion_control"]),
    ("Se requiere analista de laboratorio para ensayos y preparación de muestras", ["laboratorio"]),
    ("Buscamos encargado de bodega para control de stock y abastecimiento", ["logistica_bodega"]),
    ("Se solicita paramédico para atención de urgencias en faena", ["salud_ocupacional"]),
    ("Requerimos inspector de calidad QA QC para liberación de protocolos", ["calidad"]),
    ("Se necesita operador de grúa para maniobras de izaje en montaje", ["operador_grua"]),
    ("Vacante para cocinero de casino en campamento minero", ["alimentacion_casino"]),
    ("Se busca personal de aseo industrial y servicios generales para faena", ["aseo_servicios"]),
    ("Se requiere administrativo para gestión de documentación y acreditación de personal", ["administracion"]),
    ("Buscamos administrador de contratos para control de subcontratos", ["administracion_contratos"]),
    ("Se solicita encargado de recursos humanos para reclutamiento y remuneraciones", ["jefatura_rrhh"]),
    ("Vacante para coordinador de capacitación y desarrollo del personal", ["capacitacion_desarrollo"]),
    ("Se busca gerente de operaciones a cargo de la sucursal", ["jefatura_gerencia"]),
    ("Se requiere jefe con experiencia liderando equipos de trabajo en faena", ["liderazgo_equipos"]),
    ("Buscamos supervisor para el control y supervisión del personal a cargo", ["supervision_personal"]),
    ("Vacante para jefe de tienda en retail, reposición y atención en sala", ["retail"]),
    ("Se necesita recepcionista para atención de huéspedes en hotel", ["hoteleria"]),
    ("Se busca ejecutivo comercial para ventas y prospección de clientes", ["area_comercial_ventas"]),
    ("Requerimos profesional titulado en administración de empresas", ["administracion_empresas"]),
    ("Se solicita encargado de comercio exterior e importaciones", ["comercio_exterior"]),
    ("Vacante para analista de control de gestión e indicadores de desempeño", ["control_gestion"]),
    ("Buscamos consultor para asesoría estratégica a empresas", ["asesoria_consultoria"]),
    ("Se requiere ejecutivo de cobranza para recuperación de cartera", ["cobranza"]),
    ("Cobrador de deudas vencidas y seguimiento de pagos morosos", ["cobranza"]),
    ("Gestión de cobranza telefónica y negociación de deudas", ["cobranza"]),
    ("Recuperación de cartera y contacto con clientes morosos", ["cobranza"]),
    ("Se necesita planificador de proyectos con manejo de cronogramas", ["planificacion_proyectos"]),
    ("Vacante para ingeniero en automatización y control industrial", ["ingenieria"]),

    # ---------- REFUERZO: SUPERVISOR / LIDERAZGO EN MINA ----------
    ("Supervisor de operaciones mina a cargo de cuadrillas en terreno", ["supervision_personal", "liderazgo_equipos"]),
    ("Jefe de turno coordinando equipos operacionales en faena minera", ["supervision_personal", "liderazgo_equipos"]),
    ("Liderazgo de cuadrillas y control de producción diaria en mina", ["liderazgo_equipos", "supervision_personal"]),
    ("Coordinación de equipos operacionales y reportabilidad de turno", ["supervision_personal"]),
    ("Supervisor de terreno con foco en la seguridad y cumplimiento de estándares", ["supervision_personal"]),
    ("Ingeniero de ejecución en minas supervisando operaciones de rajo abierto", ["supervision_personal", "liderazgo_equipos"]),
    ("Jefe de operaciones liderando personal propio y contratistas en faena", ["liderazgo_equipos", "supervision_personal"]),

    # ---------- CONTRASTE: "maquinaria" que NO es de operador ----------
    # (enseñan que administrar/supervisar/mantener maquinaria NO es operarla)
    ("Administrar EPP, insumos y maquinaria del área en faena", ["administracion"]),
    ("Administración de sector: personas, recursos, equipos y maquinaria", ["administracion"]),
    ("Coordinar y supervisar equipos y el uso de la maquinaria en terreno", ["supervision_personal"]),
    ("Gestión y control de herramientas, EPP y maquinaria de la faena", ["administracion"]),
    ("Encargado de administrar personas y coordinar equipos de trabajo", ["administracion", "supervision_personal"]),
    ("Persona experta en administración de personas para coordinar y supervisar equipos y administrar EPP y maquinaria", ["administracion", "supervision_personal"]),
    ("Administrar EPP y maquinaria del sector, con estudios en administración", ["administracion"]),
    ("Supervisar equipos y administrar los EPP y la maquinaria del área", ["supervision_personal", "administracion"]),
    ("Coordinar personas y administrar recursos, EPP y maquinaria en faena", ["administracion"]),
    ("Jefe de área que administra personal, presupuesto y maquinaria del sector", ["administracion", "jefatura_gerencia"]),
]


# ======================================================================
# 2) PREPARAR LOS DATOS
#    Acá aplicamos normalizar() a CADA texto de entrenamiento.
# ======================================================================
# palabras vacías: aparecen en todos lados y no dicen el oficio, las ignoramos
STOPWORDS = ["de","la","el","en","y","a","los","las","del","con","para","por","un",
             "una","su","al","lo","como","mas","más","o","e","ni","que","se","mi",
             "sin","sobre","entre","cargo","todo","toda","todos","sus"]

textos = [normalizar(par[0]) for par in ejemplos]   # textos limpios
etiquetas = [par[1] for par in ejemplos]             # etiquetas

# Convierte las etiquetas (palabras) en columnas de 0 y 1.
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(etiquetas)

# Convierte cada texto en números según qué palabras tiene y cuánto pesan.
vectorizador = TfidfVectorizer(stop_words=STOPWORDS)
X = vectorizador.fit_transform(textos)


# ======================================================================
# 3) ENTRENAR
# ======================================================================
modelo = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
modelo.fit(X, Y)
print(f"Modelo entrenado con {len(ejemplos)} ejemplos.\n")


# ======================================================================
# 3.5) FUNCIÓN REUTILIZABLE: pedirle a Clarita las etiquetas de un texto.
#      La usa perfiles.py para analizar perfiles y hacer el match.
# ======================================================================
def clasificar_texto(texto, umbral=0.30):
    X = vectorizador.transform([normalizar(texto)])
    probs = modelo.predict_proba(X)[0]
    ranking = sorted(zip(mlb.classes_, probs), key=lambda x: x[1], reverse=True)
    elegidas = [c for c, p in ranking if p >= umbral] or [ranking[0][0]]
    return set(elegidas)


if __name__ == "__main__":
    # ======================================================================
    # 4) PROBARLO con textos NUEVOS
    #    Escribe el texto normal (con tildes y mayúsculas si quieres);
    #    normalizar() se encarga de limpiarlo igual que los ejemplos.
    # ======================================================================
    textos_nuevos = [
        "He participado en multiples proyectos industriales los cuales me han permitido "
        "conocer y aprender varios sistemas de gestion, estandares y criterios en la toma "
        "de decisiones, lo mencionado me ha entregado herramientas para contextualizar, "
        "empatizar y liderar equipos de trabajos en los cuales he sido participe.",
    ]

    # limpiamos los textos nuevos igual que los de entrenamiento
    textos_nuevos_limpios = [normalizar(t) for t in textos_nuevos]

    X_nuevo = vectorizador.transform(textos_nuevos_limpios)
    probabilidades = modelo.predict_proba(X_nuevo)
    clases = mlb.classes_

    # mostramos el texto ORIGINAL (no el limpio), para que se lea bien
    for texto, probs in zip(textos_nuevos, probabilidades):
        print(f'TEXTO: "{texto}"')

        # ordenamos las etiquetas de mayor a menor "seguridad" del modelo
        ranking = sorted(zip(clases, probs), key=lambda x: x[1], reverse=True)

        # elegimos las que superan el 40% de seguridad (si ninguna, la mejor)
        elegidas = [c for c, p in ranking if p >= 0.30] or [ranking[0][0]]

        # mostramos solo el top 6 para no llenar la pantalla
        for etiqueta, p in ranking[:6]:
            marca = "  <== etiqueta asignada" if etiqueta in elegidas else ""
            print(f"   {etiqueta:22s} {p*100:5.1f}%{marca}")
        print()
