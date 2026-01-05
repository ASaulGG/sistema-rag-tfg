"""
ingest.py

Pipeline de ingesta para un sistema RAG orientado a TFG:
- Fuente de verdad: una base de datos en Notion (items con Procesado=False).
- Extracción multimodal:
  - PDFs: LlamaParse (devuelve Markdown, incluyendo tablas cuando es posible).
  - Web: trafilatura (extrae contenido principal en Markdown).
  - YouTube: descarga de audio + transcripción con Whisper.
  - Imágenes (png/jpg/jpeg/svg): análisis con Gemini (visión o, en fallback, análisis del SVG como texto).
- Persistencia: ChromaDB como vector store (LlamaIndex + ChromaVectorStore).
"""

import os
import shutil
import requests
from dotenv import load_dotenv
from notion_client import Client
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb
import yt_dlp
import whisper
import sys
import trafilatura

from urllib.parse import urlparse, unquote, parse_qs, urlencode, urlunparse


# =============================================================================
# 1) CONFIGURACIÓN
# =============================================================================

# Carga variables desde .env (GOOGLE_API_KEY, NOTION_TOKEN, NOTION_DATABASE_ID, etc.).
load_dotenv()

# Embeddings: modelo de Google (GenAI Embeddings).
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY")
)

# LLM base de LlamaIndex.
Settings.llm = GoogleGenAI(
    model="gemini-3-pro-preview", api_key=os.getenv("GOOGLE_API_KEY")
)


# =============================================================================
# UTILIDAD: BARRA DE PROGRESO EN CONSOLA
# =============================================================================

def barra_progreso(paso, total, prefix=""):
    """Imprime una barra de progreso simple (20 bloques) en la misma línea."""
    porcentaje = int((paso / total) * 100)
    bloques = int(porcentaje / 5)
    barra = "█" * bloques + "-" * (20 - bloques)
    sys.stdout.write(f"\r{prefix} [{barra}] {porcentaje}%")
    sys.stdout.flush()
    if paso == total:
        print()


# =============================================================================
# 2) FUNCIONES DE PROCESAMIENTO / EXTRACCIÓN
# =============================================================================

def _normalize_youtube_url(url: str) -> str:
    """
    Normaliza URLs de YouTube para evitar que yt-dlp trate un vídeo como playlist
    cuando el enlace incluye parámetros tipo &list=... .
    """
    try:
        u = (url or "").strip()
        if not u:
            return u

        p = urlparse(u)

        # Formato corto: youtu.be/<id>
        if p.netloc.lower().endswith("youtu.be"):
            vid = (p.path or "").strip("/").split("/")[0]
            if not vid:
                return u
            qs = parse_qs(p.query or "")
            keep = {}
            for k in ("t", "start"):
                if k in qs and qs[k]:
                    keep[k] = qs[k][0]
            q = ("?" + urlencode(keep)) if keep else ""
            return f"https://www.youtube.com/watch?v={vid}{q}"

        # Formato estándar: youtube.com/watch?v=<id>&list=...
        qs = parse_qs(p.query or "")
        vid = (qs.get("v") or [""])[0]
        if not vid:
            return u

        keep = {"v": vid}
        for k in ("t", "start"):
            if k in qs and qs[k]:
                keep[k] = qs[k][0]

        new_p = p._replace(path="/watch", params="", query=urlencode(keep), fragment="")
        return urlunparse(new_p)

    except Exception:
        # Si algo falla, devuelve la URL original sin romper el flujo.
        return url


def descargar_youtube(url):
    """Descarga el audio de un vídeo de YouTube y lo guarda como MP3 temporal."""
    print("   🎥 Descargando audio de YouTube...")
    try:
        url = _normalize_youtube_url(url)

        # Opciones pensadas para redes “caprichosas”:
        # - noplaylist: evita bajar playlists por error
        # - timeouts/retries: reduce bloqueos indefinidos
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "outtmpl": "temp_audio.%(ext)s",
            "quiet": True,
            "overwrites": True,
            "source_address": "0.0.0.0",
            "noplaylist": True,
            "extractor_args": {"youtube": {"player_client": ["default"]}},
            "socket_timeout": 30,
            "retries": 5,
            "fragment_retries": 5,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "temp_audio.mp3"
    except Exception as e:
        print(f"   ❌ Error descargando video: {e}")
        return None


def transcribir_whisper(ruta_audio):
    """Transcribe un audio con Whisper (modelo 'base') y devuelve texto plano."""
    print("   🗣️  Transcribiendo audio (esto toma tiempo)...")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(ruta_audio, fp16=False)
        return result["text"]
    except Exception as e:
        print(f"   ❌ Error en Whisper: {e}")
        return ""


def procesar_pdf(ruta_o_url, es_url=True):
    """
    Procesa un PDF (local o por URL) con LlamaParse y devuelve el resultado en Markdown.
    Se usa un fichero temporal para unificar el flujo.
    """
    print("   📄 Analizando PDF y tablas con LlamaParse...")
    target_file = "temp.pdf"
    try:
        if es_url:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(ruta_o_url, headers=headers, stream=True)
            if response.status_code == 200:
                with open(target_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"   ❌ Error descargando PDF: Status {response.status_code}")
                return None
        else:
            shutil.copy(ruta_o_url, target_file)

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=False,
            base_url="https://api.cloud.eu.llamaindex.ai",
        )

        documents = parser.load_data(target_file)
        return "\n\n".join([doc.text for doc in documents])

    except Exception as e:
        print(f"   ❌ Error procesando PDF: {e}")
        return None
    finally:
        # Limpieza defensiva del temporal.
        if os.path.exists("temp.pdf"):
            try:
                os.remove("temp.pdf")
            except:
                pass


def extraer_contenido_web(url: str) -> str:
    """Descarga una página y extrae el contenido principal en formato Markdown."""
    print("   🌐 Descargando y extrayendo contenido de la web...")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=20)

        if resp.status_code != 200:
            print(f"   ❌ Error HTTP {resp.status_code} al descargar la web.")
            return ""

        html = resp.text
        text = trafilatura.extract(html, output_format="markdown", url=url)

        if not text:
            print("   ❌ No se ha podido extraer contenido estructurado de la web.")
            return ""

        print("   ✅ Contenido de la web extraído correctamente.")
        return text

    except Exception as e:
        print(f"   ❌ Error extrayendo contenido web: {e}")
        return ""


# =============================================================================
# 2.1) IMÁGENES: DESCARGA + RUTEADO (PNG/JPG/JPEG/SVG)
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg"}


def _guess_ext(file_name: str, url: str) -> str:
    """Obtiene una extensión fiable a partir del nombre y/o la URL (sin querystring)."""
    name = (file_name or "").strip()
    if name and "." in name:
        ext = os.path.splitext(name)[1].lower()
        if ext:
            return ext

    try:
        path = unquote(urlparse(url).path or "")
        ext = os.path.splitext(path)[1].lower()
        return ext or ""
    except Exception:
        return ""


def _download_bytes(url: str) -> bytes:
    """Descarga binarios (imágenes) desde una URL y devuelve los bytes."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content


def _extract_text_from_genai_response(resp) -> str:
    """
    Extrae texto del response del SDK 'google-genai' de forma tolerante,
    ya que la estructura del objeto puede variar según versión/SDK.
    """
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    try:
        cands = getattr(resp, "candidates", None) or []
        if cands:
            content = getattr(cands[0], "content", None)
            parts = getattr(content, "parts", None) or []
            texts = []
            for p in parts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt.strip():
                    texts.append(pt.strip())
            if texts:
                return "\n".join(texts).strip()
    except Exception:
        pass

    return str(resp).strip()


def _gemini_vision_analyze(image_bytes: bytes, mime_type: str) -> str:
    """
    Analiza una imagen con Gemini (visión) y devuelve un Markdown estructurado.
    Migrado a google.genai (google-genai). Ya no usa google.generativeai.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el entorno.")

    prompt = """Analiza esta imagen para un sistema RAG de un TFG.

Devuelve SIEMPRE en Markdown y con esta estructura:

## OCR (texto detectado)
- Si no hay texto, di: "No se detecta texto legible."

## Descripción de la imagen
- Qué aparece y qué representa (si es un diagrama, explica el flujo).

## Información útil / conceptos clave
- Lista de conceptos, definiciones, relaciones o pasos (si aplica).

## Etiquetas sugeridas
- 5 a 10 tags cortas.

Sé preciso y no inventes datos.
""".strip()

    # Permite override desde .env si quieres cambiar de modelo sin tocar el código.
    model_name = os.getenv("GEMINI_VISION_MODEL", "gemini-3-pro-image-preview")

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar google.genai.\n"
            "Instala/actualiza el SDK:\n"
            "  pip install -U google-genai\n"
            f"Detalle: {e}"
        ) from e

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    resp = client.models.generate_content(
        model=model_name,
        contents=[image_part, prompt],
    )
    return _extract_text_from_genai_response(resp)


def _gemini_text_analyze(text: str) -> str:
    """
    Llamada auxiliar a Gemini en modo texto (útil como fallback para SVG u otros casos).
    Migrado a google.genai (google-genai). Ya no usa google.generativeai.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el entorno.")
    model_name = os.getenv("GEMINI_TEXT_MODEL", "gemini-3-flash-preview")

    try:
        from google import genai
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar google.genai.\n"
            "Instala/actualiza el SDK:\n"
            "  pip install -U google-genai\n"
            f"Detalle: {e}"
        ) from e

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model_name, contents=text)
    return _extract_text_from_genai_response(resp)


def procesar_imagen(url_imagen: str, file_name: str = "") -> str:
    """
    Descarga una imagen y la analiza. Caso especial: SVG.
    - Si se puede, convierte SVG a PNG y usa visión.
    - Si no, analiza el XML del SVG como texto (fallback).
    """
    ext = _guess_ext(file_name, url_imagen)
    data = _download_bytes(url_imagen)

    if ext == ".svg":
        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(bytestring=data)
            return _gemini_vision_analyze(png_bytes, "image/png")
        except Exception:
            try:
                svg_text = data.decode("utf-8", errors="replace")
                prompt = f"""Analiza este SVG (XML) para un sistema RAG de un TFG. Devuelve Markdown con:
- Qué representa el diagrama/imagen
- Texto visible si lo hay (en <text> u otros)
- Conceptos clave
- Etiquetas sugeridas

SVG:
```xml
{svg_text}
```
""".strip()
                return _gemini_text_analyze(prompt)
            except Exception as e:
                print(f"   ❌ Error analizando SVG: {e}")
                return ""

    if ext in (".jpg", ".jpeg"):
        return _gemini_vision_analyze(data, "image/jpeg")
    if ext == ".png":
        return _gemini_vision_analyze(data, "image/png")

    # Si llega aquí, es una extensión no contemplada en IMAGE_EXTS (o sin extensión).
    return ""


# =============================================================================
# 2.2) NOTION: UTILIDADES DE VISUALIZACIÓN / EXTRACCIÓN DE CAMPOS
# =============================================================================

def extraer_titulo(props):
    """Detecta automáticamente la propiedad 'title' de una página de Notion."""
    for key, value in props.items():
        if value.get("type") == "title":
            raw = value.get("title", [])
            if raw:
                return raw[0].get("plain_text", "Sin título")
            return "Sin título"
    return "Sin título"


def imprimir_resumen_bonito(paginas):
    """Muestra un resumen legible de los items que están pendientes de procesar."""
    print("\n=========== RESUMEN DE PÁGINAS EN NOTION ===========\n")

    for page in paginas:
        props = page["properties"]

        titulo = extraer_titulo(props)
        url = props.get("URL", {}).get("url") or "—"

        archivos = props.get("File", {}).get("files", [])
        if archivos:
            archivo_obj = archivos[0]
            archivo_url = archivo_obj.get("file", {}).get("url") or archivo_obj.get(
                "external", {}
            ).get("url")
        else:
            archivo_url = "—"

        tags = props.get("Tags", {}).get("multi_select", [])
        tag_names = ", ".join([t["name"] for t in tags]) if tags else "—"

        nota = "—"
        rich = props.get("Note", {}).get("rich_text", [])
        if rich:
            nota = rich[0].get("plain_text", "—")

        like = props.get("Like", {}).get("checkbox")
        procesado = props.get("Procesado", {}).get("checkbox")

        print("────────────────────────────────────────────")
        print(f"📌 Título:        {titulo}")
        print(f"🔗 URL web:       {url}")
        print(f"📄 Archivo PDF:   {archivo_url}")
        print(f"🏷️ Tags:          {tag_names}")
        print(f"📝 Nota:          {nota}")
        print(f"👍 Like:          {like}")
        print(f"⚙️ Procesado:     {procesado}")
        print(f"🆔 ID Notion:     {page['id']}")
        print("────────────────────────────────────────────\n")

    print("=========== FIN DEL RESUMEN ===========\n")


# =============================================================================
# 3) LÓGICA PRINCIPAL
# =============================================================================

def main():
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        db_id = os.getenv("NOTION_DATABASE_ID")

        print("🔍 Consultando base de datos Notion...")

        # Solo procesamos los items que aún no están marcados como “Procesado”.
        response = notion.databases.query(
            database_id=db_id,
            filter={"property": "Procesado", "checkbox": {"equals": False}},
        )

        paginas = response.get("results", [])
        print(f"📦 Encontrados {len(paginas)} elementos nuevos para procesar.")

        imprimir_resumen_bonito(paginas)

        documentos_nuevos = []

        for p in paginas:
            props = p["properties"]
            page_id = p["id"]
            titulo = extraer_titulo(props)

            print(f"\n⚡ Procesando: {titulo}")

            texto_final = ""
            origen = ""
            tipo = "otro"

            # Barra de progreso “cosmética” (no influye en la lógica).
            total_pasos = 5
            paso = 1
            barra_progreso(paso, total_pasos, prefix="⏳ Preparando...")

            archivos = props.get("File", {}).get("files", [])
            url_web = props.get("URL", {}).get("url")
            paso += 1
            barra_progreso(paso, total_pasos, prefix="🔍 Detectando origen...")

            # Prioridad: si hay archivo adjunto, se procesa el archivo.
            if archivos:
                archivo_obj = archivos[0]
                file_name = archivo_obj.get("name", "") or ""
                url_archivo = archivo_obj.get("file", {}).get("url") or archivo_obj.get(
                    "external", {}
                ).get("url")

                ext = _guess_ext(file_name, url_archivo or "")

                if ext in IMAGE_EXTS:
                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="🖼️ Analizando imagen...")
                    texto_final = procesar_imagen(url_archivo, file_name=file_name)
                    origen = (
                        f"Imagen adjunta ({file_name})"
                        if file_name
                        else "Imagen adjunta"
                    )
                    tipo = "imagen"
                else:
                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="📄 Descargando PDF...")
                    texto_final = procesar_pdf(url_archivo, es_url=True)
                    origen = "PDF adjunto"
                    tipo = "pdf"

            # Si no hay archivo, se intenta procesar la URL.
            elif url_web:
                origen = url_web

                if "youtube.com" in url_web or "youtu.be" in url_web:
                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="🎥 Descargando vídeo...")
                    audio_path = descargar_youtube(url_web)

                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="🗣️ Transcribiendo...")

                    if audio_path and os.path.exists(audio_path):
                        texto_final = transcribir_whisper(audio_path)
                        tipo = "video"
                        # Limpieza del audio temporal.
                        try:
                            os.remove(audio_path)
                        except:
                            pass

                elif url_web.endswith(".pdf"):
                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="📄 Procesando PDF...")
                    texto_final = procesar_pdf(url_web, es_url=True)
                    tipo = "pdf"

                else:
                    paso += 1
                    barra_progreso(paso, total_pasos, prefix="🌐 Analizando web...")
                    texto_final = extraer_contenido_web(url_web)
                    tipo = "web"

                    # Si falla la extracción, al menos guardamos algo útil.
                    if not texto_final:
                        texto_final = (
                            "No se ha podido extraer el contenido. "
                            f"Se guarda su URL: {url_web}"
                        )

            barra_progreso(total_pasos, total_pasos, prefix="✅ Finalizado")

            # Solo guardamos si hemos conseguido contenido.
            if texto_final:
                doc = Document(
                    text=texto_final,
                    metadata={
                        "titulo": titulo,
                        "origen": origen,
                        "tipo": tipo,
                        "notion_id": page_id,
                    },
                )
                documentos_nuevos.append(doc)

                # Marcamos el item como procesado en Notion para no re-ingerirlo.
                try:
                    notion.pages.update(
                        page_id=page_id, properties={"Procesado": {"checkbox": True}}
                    )
                    print("   ✅ Procesado y marcado en Notion.")
                except Exception as e:
                    print(f"   ⚠️ Error marcando check en Notion: {e}")
            else:
                print("   ⚠️ No se pudo extraer contenido válido.")

        # Persistencia en ChromaDB (solo si hay documentos nuevos).
        if documentos_nuevos:
            print("\n🧠 Guardando en el cerebro (ChromaDB)...")
            db = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./cerebro_db"))
            collection = db.get_or_create_collection(
                os.getenv("COLLECTION_NAME", "tfg_master")
            )
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            VectorStoreIndex.from_documents(
                documentos_nuevos, storage_context=storage_context, show_progress=True
            )
            print("🎉 ¡Proceso terminado con éxito!")
        else:
            print("\n💤 Nada nuevo que guardar.")

    except Exception as e:
        print(f"\n❌ Error General: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
