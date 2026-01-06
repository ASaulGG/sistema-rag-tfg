---
name: "🐛 Reporte de bug"
about: "Informa un fallo reproducible (ingest / chat / deduplicación)"
title: "[BUG] "
labels: ["bug"]
assignees: []
---

## ✅ Descripción
Explica el problema en 1–3 frases.

## 🔁 Pasos para reproducir
1.
2.
3.

## ✅ Resultado esperado
¿Qué debería ocurrir?

## ❌ Resultado actual
¿Qué ocurre realmente? (incluye el mensaje de error si aplica)

## 🧪 Comando ejecutado
Marca uno y pega el comando exacto:

- [ ] `python ingest.py`
- [ ] `python chat_ui.py`
- [ ] `python check_cerebro_db_duplicates.py`

Comando exacto:
```powershell
# pega aquí el comando tal cual lo ejecutaste
```

## 🧾 Logs / traceback
Pega la salida completa (sin datos sensibles):
```text
# pega aquí
```

## 🖥️ Entorno
- SO: Windows 10 / Windows 11
- Python: (pega `python --version`)
- Método de instalación: `pip` / `uv`
- `ffmpeg -version` (si aplica):

## 🔐 Configuración (sin secretos)
⚠️ No pegues tu `.env` ni tokens.

Indica solo lo necesario:
- `CHROMA_PATH`: 
- `COLLECTION_NAME`:
- Fuente afectada: PDF / Web / YouTube / Imagen
- ¿Notion habilitado?: Sí/No

## 📎 Contexto adicional
Cualquier detalle extra, capturas o hipótesis.
