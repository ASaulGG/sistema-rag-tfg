# Plantilla de base de datos de Notion (mínimo) + Save to Notion

Este documento describe una **estructura mínima recomendada** para la base de datos de Notion y una configuración práctica de la extensión **Save to Notion** para capturar recursos desde el navegador y guardarlos con propiedades consistentes.

## 1) Propiedades recomendadas en la DB

### Mínimas
- **Title** (*title*): nombre del recurso (el nombre puede ser cualquiera; es una propiedad de tipo *title*).
- `Procesado` (*checkbox*): indicador de si el recurso ya ha sido ingerido.

### Útiles (según el tipo de fuente)
- `URL` (*url*): enlaces a páginas web.
- `File` (*files*): PDFs u otros adjuntos.

### Opcionales (calidad / organización)
- `Tags` (*multi_select*): etiquetas para clasificar recursos.
- `Like` (*checkbox*): marca rápida.
- `Saved` (*date*): fecha de guardado.

## 2) Configurar “Save to Notion” (Form de guardado)

Objetivo: definir un **formulario (Form)** para que la extensión guarde páginas/vídeos con las propiedades deseadas en la base de datos.

Pasos:
1. Abrir la extensión **Save to Notion** y pulsar **Add New Form**.
2. Seleccionar **Add to existing page/database**.
3. Seleccionar el **workspace** y la **base de datos** de Notion.
4. Asignar un nombre al formulario (ej. `Plantilla RAG-TFG`).
5. En **Fields**, revisar/editar qué propiedades se guardarán con cada recurso:
   - Para añadir propiedades: **Add New Field** o menú de los **6 puntos** → *Pick another field*.
   - En la lista aparecen *Database properties* (propiedades existentes en la DB).

Ejemplo típico de campos usados en el formulario (pueden variar según preferencia):
- **Title** → *Page Title*
- `URL` → *Url*
- `Procesado` → *(checkbox)*
- `File` → *(files)*
- `Tags` → *(multi_select)*
- `Like` → *(checkbox)*
- `Saved` → *(date)*

6. Guardar con **Save Form**.

## 3) Añadir un recurso desde el navegador

1. Abrir el recurso (web/vídeo/etc.).
2. Abrir **Save to Notion** y seleccionar el formulario creado.
3. Completar propiedades que queden vacías (p. ej. `Tags`).
4. Pulsar **Save page**.

Notas:
- Si se desea, también puede crearse la página en Notion manualmente (subir PDFs/imágenes/vídeos locales) y completar propiedades a mano.
- Algunas webs pueden restringir la extracción de contenido desde herramientas externas; en ese caso, durante la ingesta puede llegar a guardarse únicamente el **título** del recurso.
