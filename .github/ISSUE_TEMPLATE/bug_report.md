---
name: "🐛 Reporte de bug"
about: "Informar un fallo reproducible (ingest / chat / deduplicación)"
title: "[BUG] "
labels: ["bug"]
assignees: []
---

## ✅ Descripción
Describir el problema en 1–3 frases.

## 🔁 Pasos para reproducir
1.
2.
3.

## ✅ Resultado esperado
Indicar el comportamiento esperado.

## ❌ Resultado actual
Indicar el comportamiento observado (incluir el mensaje de error si aplica).

## 🧪 Comando ejecutado
Marcar una opción e incluir el comando exacto:

- [ ] `python ingest.py`
- [ ] `python chat_ui.py`
- [ ] `python check_cerebro_db_duplicates.py`

Comando exacto:
```powershell
# incluir aquí el comando tal y como se ejecutó
```

## 🧾 Logs / traceback
Incluir la salida completa (sin datos sensibles):
```text
# incluir aquí
```

## 🖥️ Entorno
- SO: Windows 10 / Windows 11
- Python: (incluir salida de `python --version`)
- Método de instalación: `pip` / `uv`
- `ffmpeg -version` (si aplica):

## 🔐 Configuración (sin secretos)
⚠️ No incluir `.env` ni tokens.

Indicar únicamente lo necesario:
- `CHROMA_PATH`:
- `COLLECTION_NAME`:
- Fuente afectada: PDF / Web / YouTube / Imagen
- ¿Notion habilitado?: Sí / No

## 📎 Contexto adicional
Añadir detalles, enlaces, capturas o hipótesis relevantes.
