"""
check_cerebro_db_duplicates.py

Utilidad para auditar una colección ChromaDB y detectar:
1) Duplicados exactos: mismo recurso (source_id) + mismo contenido (sha256 del texto normalizado).
2) Re-ingestas “reales” (heurística conservadora):
   - En recursos no-imagen: exige al menos 2 grupos de ingesta “válidos” y alto solape de contenido
     (SequenceMatcher + simhash). Además, intenta escoger automáticamente la clave de agrupación
     más fiable entre varios candidatos (doc_group_id, ref_doc_id, document_id, etc.).
   - En recursos imagen: si un mismo source_id tiene >1 ítem, se considera re-ingesta
     (habitualmente una imagen genera un único documento/ítem).

Acciones opcionales interactivas:
- Eliminar duplicados exactos dejando 1 elemento por grupo.
- Eliminar re-ingestas detectadas dejando 1 ingesta por recurso.
- Borrar un recurso completo por título y (opcionalmente) marcar su propiedad de Notion como
  Procesado=False para permitir re-procesado.

Variables de entorno:
- CHROMA_PATH, COLLECTION_NAME
- BATCH_SIZE, DELETE_BATCH
- TITLE_FILTER="..."       Filtra el análisis por título (contiene / case-insensitive)
- SHOW_SOURCE_ID="1"       Muestra el source_id en los listados
- NOTION_TOKEN             Solo necesario si se usa el borrado por título con reset en Notion
- NOTION_PROCESADO_PROP    Nombre de la propiedad checkbox en Notion (por defecto: "Procesado")
- VACUUM_AFTER_DELETE="1"  Ejecuta VACUUM automáticamente al final si hubo borrados

Umbrales de re-ingesta (conservadores):
- MIN_GROUP_SIZE_NON_IMAGE=3
- MIN_GROUP_SIZE_NON_IMAGE_SMALL=2
- RATIO_STRICT=0.94
- RATIO_LOOSE=0.90
- SIMHASH_MAX_DIST=10
"""

import os
import re
import math
import hashlib
import difflib
import sqlite3
import time
import gc
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./cerebro_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tfg_master")

# Notion: solo se usa al borrar un recurso completo por título (reset de "Procesado").
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_PROCESADO_PROP = os.getenv("NOTION_PROCESADO_PROP", "Procesado")

# Tamaños de lote para lecturas y borrados (seguridad/rendimiento).
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
DELETE_BATCH = int(os.getenv("DELETE_BATCH", "500"))

# Compactación física opcional del sqlite (reduce tamaño del fichero tras borrados).
# - Si VACUUM_AFTER_DELETE=1: ejecuta VACUUM automáticamente si hubo borrados.
# - Si no, se preguntará al final si se desea compactar.
VACUUM_AFTER_DELETE = (os.getenv("VACUUM_AFTER_DELETE", "") or "").strip().lower() in (
    "1", "true", "t", "yes", "y", "si", "sí", "s"
)

# Filtros/flags de ejecución.
TITLE_FILTER = (os.getenv("TITLE_FILTER", "") or "").strip()
SHOW_SOURCE_ID = (os.getenv("SHOW_SOURCE_ID", "") or "").strip() in (
    "1", "true", "True", "yes", "y", "si", "sí", "S"
)

# Umbrales de re-ingesta (modo conservador).
MIN_GROUP_SIZE_NON_IMAGE = int(os.getenv("MIN_GROUP_SIZE_NON_IMAGE", "3"))
MIN_GROUP_SIZE_NON_IMAGE_SMALL = int(os.getenv("MIN_GROUP_SIZE_NON_IMAGE_SMALL", "2"))
RATIO_STRICT = float(os.getenv("RATIO_STRICT", "0.94"))
RATIO_LOOSE = float(os.getenv("RATIO_LOOSE", "0.90"))
SIMHASH_MAX_DIST = int(os.getenv("SIMHASH_MAX_DIST", "10"))


def _s(meta: Optional[Dict[str, Any]], *keys: str) -> str:
    """Obtiene el primer valor “usable” (no vacío) de meta para una lista de claves."""
    if not isinstance(meta, dict):
        return ""
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            v = str(v).strip()
            if v:
                return v
        else:
            v = str(v).strip()
            if v:
                return v
    return ""


def _norm_text(t: Any) -> str:
    """Normaliza saltos de línea y espacios para estabilizar el hashing y comparaciones."""
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join([re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")])
    return t.strip()


def _sha256(text: str) -> str:
    """Hash sha256 del texto (UTF-8, ignorando errores de codificación)."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _pretty(meta: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Devuelve una tupla (titulo, origen, tipo) con fallbacks habituales."""
    titulo = _s(meta, "titulo", "title", "file_name", "filename") or "Sin título"
    origen = _s(meta, "origen", "url", "source", "file_path", "path") or "—"
    tipo = _s(meta, "tipo", "type") or "—"
    return titulo, origen, tipo


def _source_id(meta: Optional[Dict[str, Any]]) -> str:
    """
    Identificador estable de recurso para agrupar ítems de la colección.

    Prioriza (si existe):
    1) notion_id
    2) origen/url/path
    3) título
    """
    nid = _s(meta, "notion_id", "notionId", "notion_page_id")
    if nid:
        return f"notion_id:{nid}"
    org = _s(meta, "origen", "url", "source", "file_path", "path")
    if org:
        return f"origen:{org}"
    tit = _s(meta, "titulo", "title", "file_name", "filename")
    if tit:
        return f"titulo:{tit}"
    return "desconocido"


def _fetch_all(
    collection,
) -> Tuple[List[str], List[Optional[Dict[str, Any]]], List[str]]:
    """
    Recupera todos los ids, metadatos y documentos de la colección.

    Soporta dos modos:
    - Paginación (limit/offset) cuando la API lo permite.
    - Lectura completa en una llamada cuando no hay soporte de paginación.
    """
    total = collection.count()
    if total <= 0:
        return [], [], []

    ids_all: List[str] = []
    metas_all: List[Optional[Dict[str, Any]]] = []
    docs_all: List[str] = []

    supports_paging = True
    try:
        collection.get(limit=1, offset=0, include=["metadatas"])
    except TypeError:
        supports_paging = False
    except Exception:
        supports_paging = True

    if not supports_paging:
        res = collection.get(include=["metadatas", "documents"])
        ids_all = list(res.get("ids") or [])
        metas_all = list(res.get("metadatas") or [])
        docs_all = [_norm_text(x) for x in (res.get("documents") or [])]
        return ids_all, metas_all, docs_all

    pages = int(math.ceil(total / BATCH_SIZE))
    for p in range(pages):
        offset = p * BATCH_SIZE
        res = collection.get(
            include=["metadatas", "documents"],
            limit=BATCH_SIZE,  # type: ignore
            offset=offset,  # type: ignore
        )
        ids = list(res.get("ids") or [])
        metas = list(res.get("metadatas") or [])
        docs = [_norm_text(x) for x in (res.get("documents") or [])]

        ids_all.extend(ids)
        metas_all.extend(metas)
        docs_all.extend(docs)

    return ids_all, metas_all, docs_all


def _delete_ids(collection, ids_to_delete: List[str]) -> None:
    """Borra ids en lotes para evitar peticiones demasiado grandes."""
    if not ids_to_delete:
        return
    for k in range(0, len(ids_to_delete), DELETE_BATCH):
        batch = ids_to_delete[k : k + DELETE_BATCH]
        collection.delete(ids=batch)


def _locate_chroma_sqlite(chroma_path: str) -> Optional[str]:
    """Localiza el fichero sqlite de Chroma en CHROMA_PATH."""
    direct = os.path.join(chroma_path, "chroma.sqlite3")
    if os.path.isfile(direct):
        return direct
    # Fallback: primer .sqlite3 dentro del directorio.
    try:
        cands = [
            os.path.join(chroma_path, f)
            for f in os.listdir(chroma_path)
            if f.lower().endswith(".sqlite3")
        ]
    except Exception:
        return None
    for p in sorted(cands):
        if os.path.isfile(p):
            return p
    return None


def _stop_chroma_client(client: Any) -> None:
    """Intenta cerrar/parar el sistema interno de Chroma para liberar locks del sqlite."""
    try:
        sys_obj = getattr(client, "_system", None)
        if sys_obj is not None and hasattr(sys_obj, "stop"):
            sys_obj.stop()
    except Exception:
        pass
    try:
        if hasattr(client, "close"):
            client.close()  # type: ignore[call-arg]
    except Exception:
        pass


def _vacuum_sqlite(db_file: str, retries: int = 3) -> bool:
    """Ejecuta VACUUM para compactar el sqlite (reclama espacio en disco)."""
    if not db_file or not os.path.isfile(db_file):
        return False

    for attempt in range(1, retries + 1):
        try:
            con = sqlite3.connect(db_file, timeout=60)
            try:
                con.execute("PRAGMA busy_timeout=60000;")
                con.execute("VACUUM;")
                con.commit()
            finally:
                con.close()
            return True
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if ("locked" in msg or "busy" in msg) and attempt < retries:
                time.sleep(0.6 * attempt)
                continue
            print(f"⚠️ No se pudo compactar sqlite (VACUUM). Detalle: {e}")
            return False
        except Exception as e:
            print(f"⚠️ No se pudo compactar sqlite (VACUUM). Detalle: {e}")
            return False
    return False


def _maybe_compact_db(client: Any, chroma_path: str) -> None:
    """Compacta chroma.sqlite3 si se confirma (o si VACUUM_AFTER_DELETE=1)."""
    db_file = _locate_chroma_sqlite(chroma_path)
    if not db_file:
        print("ℹ️ No se encontró chroma.sqlite3 para compactar.")
        return

    before = os.path.getsize(db_file)

    if not VACUUM_AFTER_DELETE:
        ans = (
            input(
                "\n¿Quieres COMPACTAR (VACUUM) el sqlite para reducir el tamaño en disco? [s/N]: "
            )
            .strip()
            .lower()
        )
        if ans not in ("s", "si", "sí", "y", "yes"):
            return

    # Se detiene Chroma antes del VACUUM para minimizar fallos por locks (Windows).
    _stop_chroma_client(client)
    gc.collect()
    time.sleep(0.2)

    ok = _vacuum_sqlite(db_file)
    if ok:
        after = os.path.getsize(db_file)
        print(
            f"✅ Compactación completada: {before/1024:.0f} KB → {after/1024:.0f} KB (archivo: {os.path.basename(db_file)})"
        )


_WORD_RE = re.compile(r"[a-zA-Z0-9À-ÿ_]+", re.UNICODE)


def _simhash64(text: str) -> int:
    """SimHash (64-bit) simple basado en tokens alfanuméricos."""
    tokens = _WORD_RE.findall((text or "").lower())
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = hashlib.sha1(tok.encode("utf-8", errors="ignore")).digest()
        x = int.from_bytes(h[:8], "big", signed=False)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= 1 << i
    return out


def _hamming64(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros de 64 bits."""
    return (a ^ b).bit_count()


def _rep_text(
    docs: List[str],
    limit_chars: int = 45000,
    per_doc_cap: int = 2500,
    max_docs: int = 30,
) -> str:
    """
    Construye un texto representativo de un grupo, acotando tamaño total.
    Se usa para comparaciones (ratio/simhash) sin penalizar rendimiento.
    """
    parts: List[str] = []
    total = 0
    for d in docs[:max_docs]:
        if not d:
            continue
        chunk = d[:per_doc_cap]
        parts.append(chunk)
        total += len(chunk)
        if total >= limit_chars:
            break
    return "\n".join(parts)


def _ratio(a: str, b: str) -> float:
    """Ratio de similitud aproximada (difflib.SequenceMatcher)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Claves candidatas para identificar “grupo de ingesta” en metadatos.
GROUP_KEY_CANDIDATES = (
    "doc_group_id",
    "ref_doc_id",
    "document_id",
    "source_doc_id",
    "doc_id",
    "documentId",
    "refDocId",
)


def _pick_grouping_key(
    metas: List[Optional[Dict[str, Any]]], tipo: str
) -> Optional[str]:
    """
    Selecciona la clave de agrupación más útil para separar ingestas.

    Criterio (aprox.):
    - Maximiza el tamaño del grupo mayor (evita claves por-chunk).
    - Prioriza cobertura (cuántos ítems tienen esa clave).
    - Penaliza número excesivo de grupos (ruido).
    """
    best_key = None
    best_score = None  # tuple comparable

    for key in GROUP_KEY_CANDIDATES:
        vals: List[str] = []
        for m in metas:
            v = _s(m, key)
            if v:
                vals.append(v)
        if len(vals) < 2:
            continue

        counts: Dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1

        num_groups = len(counts)
        max_group = max(counts.values()) if counts else 0
        coverage = len(vals) / max(1, len(metas))

        # En no-imagen: descartar claves que parecen “ID por chunk” (todos grupos size 1).
        if tipo != "imagen" and max_group <= 1:
            continue

        score = (max_group, coverage, -num_groups)

        if best_score is None or score > best_score:
            best_score = score
            best_key = key

    return best_key


def _collect_notion_ids(meta: Optional[Dict[str, Any]]) -> List[str]:
    """Extrae una lista de page_id(s) de Notion si existen en los metadatos."""
    nid = _s(meta, "notion_id", "notionId", "notion_page_id")
    if not nid:
        return []
    out: List[str] = []
    for part in str(nid).replace(",", " ").split():
        p = part.strip()
        if p and p != "—":
            out.append(p)
    return out


def _reset_notion_procesado(notion_ids: List[str]) -> None:
    """
    Marca Procesado=False en las páginas de Notion asociadas.
    Solo se ejecuta si se borra un recurso por título.
    """
    if not notion_ids:
        print("ℹ️ No hay notion_id en los metadatos. No se puede actualizar Notion.")
        return
    if not NOTION_TOKEN:
        print(
            "⚠️ NOTION_TOKEN no configurado. No se puede actualizar Notion (Procesado=False)."
        )
        return

    try:
        from notion_client import Client
    except Exception as e:
        print(
            f"⚠️ Falta 'notion-client'. Instala con: pip install notion-client | Detalle: {e}"
        )
        return

    notion = Client(auth=NOTION_TOKEN)
    ok = 0
    fail = 0
    for pid in sorted(set(notion_ids)):
        try:
            notion.pages.update(
                page_id=pid, properties={NOTION_PROCESADO_PROP: {"checkbox": False}}
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"⚠️ No se pudo actualizar Notion (page_id={pid}): {e}")
    if ok:
        print(f"✅ Notion actualizado: {ok} página(s) -> {NOTION_PROCESADO_PROP}=False")
    if fail and not ok:
        print("⚠️ No se pudo actualizar Notion para ninguna página.")


def _title_key(meta: Optional[Dict[str, Any]]) -> str:
    """Clave normalizada de título para indexación (case-insensitive)."""
    return (_s(meta, "titulo", "title", "file_name", "filename") or "").strip()


def _build_title_index(
    ids: List[str], metas: List[Optional[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Indexa la colección por título para permitir borrado interactivo por match exacto."""
    out: Dict[str, Dict[str, Any]] = {}
    for _id, meta in zip(ids, metas):
        t = _title_key(meta)
        if not t:
            continue
        k = t.lower()
        e = out.get(k)
        if e is None:
            titulo, origen, tipo = _pretty(meta)
            out[k] = {
                "title": titulo,
                "ids": [_id],
                "origen": origen,
                "tipo": tipo,
                "notion_ids": set(_collect_notion_ids(meta)),
            }
        else:
            e["ids"].append(_id)
            e["notion_ids"].update(_collect_notion_ids(meta))
    return out


def _prompt_delete_resource(collection) -> bool:
    """Borrado interactivo de un recurso completo por título (y reset opcional en Notion)."""
    ans = (
        input("\n¿Quieres BORRAR un recurso completo por título? [s/N]: ")
        .strip()
        .lower()
    )
    if ans not in ("s", "si", "sí", "y", "yes"):
        return False

    title_in = input(
        "Escribe el TÍTULO COMPLETO exactamente como aparece (incluye emojis si los tiene): "
    ).strip()
    if not title_in:
        print("✅ Título vacío. Cancelado.")
        return False

    ids2, metas2, _ = _fetch_all(collection)
    if not ids2:
        print("✅ La colección está vacía. Nada que borrar.")
        return False

    idx = _build_title_index(ids2, metas2)
    key = title_in.lower()

    if key not in idx:
        keys = list(idx.keys())
        close = difflib.get_close_matches(key, keys, n=8, cutoff=0.55)
        print(
            "\n❌ No se encontró un recurso con ese título (match exacto, sin distinguir mayúsculas/minúsculas)."
        )
        if close:
            print("¿Quizá querías decir uno de estos?")
            for ck in close:
                print(f"- {idx[ck]['title']}")
        else:
            subs = [k for k in keys if key in k][:8]
            if subs:
                print("Sugerencias por coincidencia parcial:")
                for sk in subs:
                    print(f"- {idx[sk]['title']}")
        return False

    info = idx[key]
    ids_to_delete = list(info["ids"])
    notion_ids = list(info.get("notion_ids") or [])

    print("\n🧾 Recurso encontrado:")
    print(f"- Título: {info['title']}")
    print(f"- Tipo:   {info['tipo']}")
    print(f"- Origen: {info['origen']}")
    print(f"- Notion page_id(s): {', '.join(notion_ids) if notion_ids else '—'}")
    print(f"- Items a borrar: {len(ids_to_delete)}")

    confirm = input(
        "Escribe BORRAR para confirmar (cualquier otra cosa cancela): "
    ).strip()
    if confirm != "BORRAR":
        print("✅ Cancelado. No se ha borrado nada.")
        return False

    print(f"\n🗑️ Borrando {len(ids_to_delete)} ids (en lotes de {DELETE_BATCH})…")
    _delete_ids(collection, ids_to_delete)
    print("✅ Recurso borrado.")

    _reset_notion_procesado(notion_ids)
    return True


def _group_indices_by_source(
    ids: List[str], metas: List[Optional[Dict[str, Any]]], docs: List[str]
) -> Dict[str, List[int]]:
    """Agrupa índices por source_id (con filtro opcional por título)."""
    out: Dict[str, List[int]] = {}
    for i, (m, d) in enumerate(zip(metas, docs)):
        if not d:
            continue
        titulo, _, _ = _pretty(m)
        if TITLE_FILTER and (TITLE_FILTER.lower() not in titulo.lower()):
            continue
        sid = _source_id(m)
        out.setdefault(sid, []).append(i)
    return out


def _tipo_majoritario(metas: List[Optional[Dict[str, Any]]]) -> str:
    """Devuelve el tipo más frecuente dentro de un grupo de metadatos."""
    counts: Dict[str, int] = {}
    for m in metas:
        t = (_s(m, "tipo", "type") or "—").strip()
        counts[t] = counts.get(t, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "—"


def _min_group_size_for(tipo: str, total_items: int) -> int:
    """Selecciona tamaño mínimo de grupo según tipo y tamaño total del recurso."""
    if tipo == "imagen":
        return 1
    if total_items <= 4:
        return MIN_GROUP_SIZE_NON_IMAGE_SMALL
    return MIN_GROUP_SIZE_NON_IMAGE


def _detect_exact_dups(
    ids: List[str], metas: List[Optional[Dict[str, Any]]], docs: List[str]
) -> Dict[Tuple[str, str], List[int]]:
    """Detecta duplicados exactos por (source_id, sha256(texto_normalizado))."""
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        if not doc:
            continue
        titulo, _, _ = _pretty(meta)
        if TITLE_FILTER and (TITLE_FILTER.lower() not in titulo.lower()):
            continue
        sid = _source_id(meta)
        h = _sha256(doc)
        groups.setdefault((sid, h), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _detect_true_reingests(
    ids: List[str], metas: List[Optional[Dict[str, Any]]], docs: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Detecta re-ingestas probables por recurso (source_id).

    En no-imagen:
    - Agrupa por la mejor clave “grupo de ingesta” disponible.
    - Filtra grupos pequeños (para evitar “chunks sueltos”).
    - Compara los dos grupos principales para decidir si son el mismo contenido.
    """
    by_source = _group_indices_by_source(ids, metas, docs)
    out: Dict[str, Dict[str, Any]] = {}

    for sid, idxs in by_source.items():
        metas_s = [metas[i] for i in idxs]
        titulo, origen, _ = _pretty(metas_s[0] if metas_s else None)

        tipo = _tipo_majoritario(metas_s)
        total_items = len(idxs)

        # Imagen: si hay >1 ítem, se considera re-ingesta (una imagen suele generar 1 ítem).
        if tipo == "imagen":
            if total_items > 1:
                out[sid] = {
                    "title": titulo,
                    "origen": origen,
                    "tipo": tipo,
                    "notion_ids": sorted(
                        set(sum((_collect_notion_ids(metas[i]) for i in idxs), []))
                    ),
                    "groups": {"(singletons)": idxs[:]},
                    "keep_gid": "(singletons)",
                    "keep_indices": [idxs[0]],
                    "delete_indices": idxs[1:],
                }
            continue

        group_key = _pick_grouping_key(metas_s, tipo=tipo)
        if not group_key:
            continue

        groups: Dict[str, List[int]] = {}
        for i in idxs:
            gid = _s(metas[i], group_key)
            if not gid:
                continue  # sin gid: no se usa (modo conservador)
            groups.setdefault(gid, []).append(i)

        if len(groups) <= 1:
            continue

        min_sz = _min_group_size_for(tipo, total_items)
        valid = {gid: is_ for gid, is_ in groups.items() if len(is_) >= min_sz}
        if len(valid) <= 1:
            continue

        ordered = sorted(valid.items(), key=lambda kv: len(kv[1]), reverse=True)
        (gid_a, ia) = ordered[0]
        (gid_b, ib) = ordered[1]

        rep_a = _rep_text([docs[i] for i in ia])
        rep_b = _rep_text([docs[i] for i in ib])

        r = _ratio(rep_a, rep_b)
        sh_a = _simhash64(rep_a)
        sh_b = _simhash64(rep_b)
        dist = _hamming64(sh_a, sh_b)

        is_true = (r >= RATIO_STRICT) or (r >= RATIO_LOOSE and dist <= SIMHASH_MAX_DIST)
        if not is_true:
            continue

        keep_gid = gid_a
        keep_indices = ia[:]
        delete_indices: List[int] = []
        for gid, is_ in valid.items():
            if gid == keep_gid:
                continue
            delete_indices.extend(is_)

        out[sid] = {
            "title": titulo,
            "origen": origen,
            "tipo": tipo,
            "group_key": group_key,
            "ratio": r,
            "simhash_dist": dist,
            "notion_ids": sorted(
                set(sum((_collect_notion_ids(metas[i]) for i in idxs), []))
            ),
            "groups": {gid: is_ for gid, is_ in valid.items()},
            "keep_gid": keep_gid,
            "keep_indices": keep_indices,
            "delete_indices": delete_indices,
        }

    return out


def main() -> int:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    did_delete = False

    print(f"🧠 Chroma path: {CHROMA_PATH}")
    print(f"📚 Collection:  {COLLECTION_NAME}")
    print(f"🔢 Total items: {collection.count()}")

    ids, metas, docs = _fetch_all(collection)
    if not ids:
        print("\n✅ La colección está vacía. Nada que comprobar.")
        did_delete = _prompt_delete_resource(collection) or did_delete
        return 0

    # 1) Duplicados exactos
    dup_groups = _detect_exact_dups(ids, metas, docs)
    if not dup_groups:
        print("\n✅ No se han encontrado duplicados EXACTOS (source_id + hash_texto).")
    else:
        total_extra = sum(len(v) - 1 for v in dup_groups.values())
        print("\n⚠️ DUPLICADOS EXACTOS DETECTADOS (source_id + hash_texto)")
        print(f"- Grupos duplicados: {len(dup_groups)}")
        print(f"- Registros extra (potenciales a borrar): {total_extra}")

        by_title: Dict[str, Dict[str, Any]] = {}
        for (sid, _h), idxs in dup_groups.items():
            i0 = idxs[0]
            titulo, origen, tipo = _pretty(metas[i0])
            entry = by_title.setdefault(
                titulo,
                {
                    "count_groups": 0,
                    "count_extra": 0,
                    "origen": origen,
                    "tipo": tipo,
                    "source_id": sid,
                },
            )
            entry["count_groups"] += 1
            entry["count_extra"] += len(idxs) - 1

        ordered = sorted(
            by_title.items(), key=lambda kv: kv[1]["count_extra"], reverse=True
        )

        print("\n📌 Títulos con duplicados EXACTOS (top):")
        for titulo, info in ordered[:40]:
            sid_txt = f" | {info['source_id']}" if SHOW_SOURCE_ID else ""
            print(
                f"- {titulo} | extra={info['count_extra']} | grupos={info['count_groups']} | "
                f"tipo={info['tipo']} | origen={info['origen']}{sid_txt}"
            )
        if len(ordered) > 40:
            print(f"… (+{len(ordered) - 40} más)")

        ans = (
            input(
                "\n¿Quieres ELIMINAR los duplicados EXACTOS y dejar 1 por grupo? [s/N]: "
            )
            .strip()
            .lower()
        )
        if ans in ("s", "si", "sí", "y", "yes"):
            confirm = input(
                "Escribe ELIMINAR para confirmar (cualquier otra cosa cancela): "
            ).strip()
            if confirm == "ELIMINAR":
                to_delete: List[str] = []
                for (_sid, _h), idxs in dup_groups.items():
                    to_delete.extend(ids[j] for j in idxs[1:])
                print(
                    f"\n🗑️ Borrando {len(to_delete)} ids (en lotes de {DELETE_BATCH})…"
                )
                _delete_ids(collection, to_delete)
                did_delete = True
                print("✅ Borrado completado (duplicados EXACTOS).")
                print(f"🔢 Total items ahora: {collection.count()}")
            else:
                print("✅ Cancelado. No se ha borrado nada.")
        else:
            print("✅ No se ha borrado nada (duplicados EXACTOS).")

    # 2) Re-ingestas “reales”
    reing = _detect_true_reingests(ids, metas, docs)
    if not reing:
        print(
            "\n✅ No se han encontrado re-ingestas VERDADERAS (heurística conservadora)."
        )
    else:
        total_extra = sum(len(v.get("delete_indices") or []) for v in reing.values())
        print("\n⚠️ RE-INGESTAS VERDADERAS DETECTADAS (conservador)")
        print(
            "   (No confunde chunks de una sola ingesta; ignora 'chunks sueltos' en no-imagen.)"
        )
        print(f"- Recursos con re-ingesta verdadera: {len(reing)}")
        print(f"- Items extra (potenciales a borrar): {total_extra}")

        ordered = sorted(
            reing.items(),
            key=lambda kv: len(kv[1].get("delete_indices") or []),
            reverse=True,
        )
        print("\n📌 Re-ingestas verdaderas (top):")
        for sid, info in ordered[:40]:
            extra = len(info.get("delete_indices") or [])
            ingestas = len(info.get("groups") or {})
            sid_txt = f" | source_id={sid}" if SHOW_SOURCE_ID else ""
            ratio_txt = ""
            if info.get("tipo") != "imagen":
                ratio_txt = (
                    f" | ratio={info.get('ratio'):.3f} | simhash={info.get('simhash_dist')}"
                )
            print(
                f"- {info['title']} | extra={extra} | ingestas={ingestas} | tipo={info['tipo']} | origen={info['origen']}{ratio_txt}{sid_txt}"
            )
        if len(ordered) > 40:
            print(f"… (+{len(ordered) - 40} más)")

        ans = (
            input(
                "\n¿Quieres ELIMINAR re-ingestas VERDADERAS y dejar 1 ingesta por recurso? [s/N]: "
            )
            .strip()
            .lower()
        )
        if ans in ("s", "si", "sí", "y", "yes"):
            confirm = input(
                "Escribe ELIMINAR para confirmar (cualquier otra cosa cancela): "
            ).strip()
            if confirm == "ELIMINAR":
                to_delete_ids: List[str] = []
                for _sid, info in reing.items():
                    to_delete_ids.extend(
                        ids[j] for j in (info.get("delete_indices") or [])
                    )
                print(
                    f"\n🗑️ Borrando {len(to_delete_ids)} ids (en lotes de {DELETE_BATCH})…"
                )
                _delete_ids(collection, to_delete_ids)
                did_delete = True
                print("✅ Borrado completado (re-ingestas VERDADERAS).")
                print(f"🔢 Total items ahora: {collection.count()}")
            else:
                print("✅ Cancelado. No se ha borrado nada.")
        else:
            print("✅ No se ha borrado nada (re-ingestas VERDADERAS).")

    # 3) Borrado por título + reset en Notion (si aplica)
    did_delete = _prompt_delete_resource(collection) or did_delete

    print(f"\n🔢 Total items final: {collection.count()}")

    # Compactación física (VACUUM): tras borrados el fichero sqlite no reduce tamaño automáticamente.
    if did_delete:
        _maybe_compact_db(client, CHROMA_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
