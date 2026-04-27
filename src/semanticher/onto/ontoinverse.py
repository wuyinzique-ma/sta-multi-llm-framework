from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import owlready2


class InversePropertyManager:
    """
    Responsibilities (aligned with your notebook cells):
      1) Read inverse_conflicts.json (pairs + inverse_map). If missing/invalid -> build from ontology
         (uri/label, label-first, fallback uri.name), optionally using ambiguous_mapping to pick stable names,
         then save to inverse_conflicts.json.
      2) After process_llm_output, detect inverse conflicts; if any, call LLM resolver once per conflict pair
         to choose the better property (winner) and remove the loser.

    inverse_conflicts.json schema:
      {
        "pairs": [["A","B"], ...],        # sorted, unique
        "inverse_map": {"A":"B","B":"A"}  # bi-directional
      }
    """

    # -------------------------- init --------------------------

    def __init__(
        self,
        rdf_path: str,
        *,
        only_local: bool = False,
        mapping_path: Optional[str] = None,  # points to .../ambiguous_mapping.json (optional)
        inverse_conflicts_path: Optional[str] = None,  # if None and mapping_path set -> sibling file
    ):
        self.rdf_path = rdf_path
        self.only_local = only_local
        self.mapping_path = mapping_path

        self.inverse_conflicts_path: Optional[Path] = None
        if inverse_conflicts_path:
            self.inverse_conflicts_path = Path(inverse_conflicts_path)
        elif mapping_path:
            mp = Path(mapping_path)
            self.inverse_conflicts_path = mp.parent / "inverse_conflicts.json"

        self.pairs: List[Tuple[str, str]] = []
        self.inverse_map: Dict[str, str] = {}

        # cache: uri -> mapped stable name (from ambiguous_mapping.json)
        self._amb_uri_to_name: Dict[str, str] = {}

    # -------------------------- robust JSON I/O (same style as OntologyLoader) --------------------------

    @staticmethod
    def _safe_read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None:
            return None
        try:
            if (not path.exists()) or path.stat().st_size == 0:
                return None
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                return None
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -------------------------- mapping (ambiguous) --------------------------

    @staticmethod
    def _load_mapping_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must be a JSON object: {path}")
        if "properties" not in data:
            raise ValueError(f"Mapping file missing 'properties' key: {path}")
        if not isinstance(data["properties"], list):
            raise ValueError(f"Mapping file 'properties' must be a list: {path}")
        return data

    def _load_ambiguous_uri_to_name(self) -> Dict[str, str]:
        if not self.mapping_path:
            return {}
        try:
            if not os.path.exists(self.mapping_path):
                return {}
            raw = self._load_mapping_file(self.mapping_path)
            out: Dict[str, str] = {}
            for e in raw.get("properties") or []:
                if not isinstance(e, dict):
                    continue
                uri = (e.get("uri") or "").strip()
                name = (e.get("name") or "").strip()
                if uri and name:
                    out[uri] = name
            return out
        except Exception:
            # mapping is optional for inverse_conflicts; if it fails, just ignore it
            return {}

    # -------------------------- OWL helpers --------------------------

    @staticmethod
    def _get_first_lang_string(values) -> Optional[str]:
        if not values:
            return None
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @classmethod
    def _get_label(cls, entity) -> Optional[str]:
        if hasattr(entity, "label"):
            s = cls._get_first_lang_string(entity.label)
            if s:
                return s
        return None

    @staticmethod
    def _uri_local_name(uri: str) -> str:
        if not isinstance(uri, str) or not uri:
            return ""
        if "#" in uri:
            return uri.rsplit("#", 1)[-1].strip()
        if "/" in uri:
            return uri.rsplit("/", 1)[-1].strip()
        return uri.strip()

    def _preferred_name(self, entity) -> str:
        """
        Name policy for inverse_conflicts:
          - if entity IRI exists in ambiguous_mapping => use mapping 'name'
          - else label-first
          - else entity.name
          - else uri local name
        """
        uri = str(getattr(entity, "iri", None) or str(entity)).strip()

        if uri and uri in self._amb_uri_to_name:
            return self._amb_uri_to_name[uri]

        lbl = self._get_label(entity)
        if lbl:
            return lbl

        nm = (getattr(entity, "name", "") or "").strip()
        if nm:
            return nm

        return self._uri_local_name(uri)

    # -------------------------- inverse_conflicts.json I/O --------------------------

    @staticmethod
    def _normalize_loaded_inverse_conflicts(data: Dict[str, Any]) -> Optional[Tuple[List[Tuple[str, str]], Dict[str, str]]]:
        pairs_raw = data.get("pairs")
        inv_raw = data.get("inverse_map")

        if not isinstance(pairs_raw, list) or not isinstance(inv_raw, dict):
            return None

        pairs: List[Tuple[str, str]] = []
        seen = set()
        for it in pairs_raw:
            if not (isinstance(it, (list, tuple)) and len(it) == 2):
                continue
            a, b = it[0], it[1]
            if not (isinstance(a, str) and isinstance(b, str)):
                continue
            a, b = a.strip(), b.strip()
            if not a or not b or a == b:
                continue
            x, y = sorted([a, b])
            if (x, y) not in seen:
                seen.add((x, y))
                pairs.append((x, y))

        inverse_map: Dict[str, str] = {}
        for k, v in inv_raw.items():
            if not (isinstance(k, str) and isinstance(v, str)):
                continue
            ks, vs = k.strip(), v.strip()
            if not ks or not vs or ks == vs:
                continue
            inverse_map[ks] = vs

        if not inverse_map:
            return None

        # optional sanity: ensure bi-directional for loaded keys when possible
        # (do not fail hard; just best-effort)
        for k, v in list(inverse_map.items()):
            if inverse_map.get(v) is None:
                inverse_map[v] = k

        pairs.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        inverse_map = dict(sorted(inverse_map.items(), key=lambda x: x[0].lower()))
        return pairs, inverse_map

    def load_inverse_conflicts(self) -> bool:
        """
        Read inverse_conflicts.json into self.pairs/self.inverse_map.
        Return True if loaded successfully, else False.
        """
        data = self._safe_read_json(self.inverse_conflicts_path)
        if not data:
            return False
        norm = self._normalize_loaded_inverse_conflicts(data)
        if not norm:
            return False
        self.pairs, self.inverse_map = norm
        return True

    def save_inverse_conflicts(self) -> None:
        """
        Save current self.pairs/self.inverse_map into inverse_conflicts.json (atomic).
        """
        if self.inverse_conflicts_path is None:
            raise ValueError("inverse_conflicts_path is not set. Provide mapping_path or inverse_conflicts_path.")
        # stable ordering
        pairs = sorted({tuple(sorted((a, b))) for a, b in self.pairs if a and b and a != b})
        inv = dict(self.inverse_map)
        inv = dict(sorted(inv.items(), key=lambda x: x[0].lower()))
        payload = {
            "pairs": [[a, b] for a, b in pairs],
            "inverse_map": inv,
        }
        self._atomic_write_json(self.inverse_conflicts_path, payload)

    # -------------------------- build from ontology (Cell 2 extraction) --------------------------

    def build_inverse_index_from_ontology(self) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """
        Build inverse pairs/map from ontology object properties using owlready2 inverse_property.
        Updates self.pairs/self.inverse_map and returns them.
        """
        onto = owlready2.get_ontology(self.rdf_path).load()

        def is_local(x) -> bool:
            return (not self.only_local) or (getattr(x, "namespace", None) == onto)

        # load mapping cache once
        self._amb_uri_to_name = self._load_ambiguous_uri_to_name()

        inverse_map: Dict[str, str] = {}
        pairs: List[Tuple[str, str]] = []
        seen = set()

        for prop in onto.object_properties():
            if not is_local(prop):
                continue
            inv = getattr(prop, "inverse_property", None)
            if inv is None or (not is_local(inv)):
                continue

            a = self._preferred_name(prop)
            b = self._preferred_name(inv)
            if not a or not b or a == b:
                continue

            inverse_map[a] = b
            inverse_map[b] = a

            x, y = sorted([a, b])
            if (x, y) not in seen:
                seen.add((x, y))
                pairs.append((x, y))

        pairs.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        inverse_map = dict(sorted(inverse_map.items(), key=lambda x: x[0].lower()))

        self.pairs = pairs
        self.inverse_map = inverse_map
        return pairs, inverse_map

    def prepare_inverse_conflicts(self) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """
        Read-first:
          - if inverse_conflicts.json loads -> use it
          - else -> build from ontology and save, then use it
        """
        loaded = self.load_inverse_conflicts()
        if loaded:
            return self.pairs, self.inverse_map

        self.build_inverse_index_from_ontology()
        self.save_inverse_conflicts()
        return self.pairs, self.inverse_map

    # -------------------------- postprocess after process_llm_output (Cell 6) --------------------------

    def attach_inverse_conflicts(self, filtered_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        filtered_results[*]["property"] = [[name, score], ...] in NAME-space.
        Adds:
          filtered_results[*]["inverse_conflicts"] = [[a,b], ...] when both appear.
        """
        inv = self.inverse_map or {}
        if not inv:
            return filtered_results

        for col in filtered_results:
            props = col.get("property", []) or []
            present = {u for u, _ in props if isinstance(u, str) and u.strip()}
            conflicts: List[List[str]] = []
            seen = set()
            for u in present:
                v = inv.get(u)
                if v and v in present:
                    a, b = sorted([u, v])
                    if (a, b) not in seen:
                        seen.add((a, b))
                        conflicts.append([a, b])
            if conflicts:
                col["inverse_conflicts"] = conflicts

        return filtered_results

    def resolve_inverse_conflicts(
        self,
        filtered_results: List[Dict[str, Any]],
        table_name: str,
        resolver_chain,
        *,
        extract_json_from_response: Callable[[str], str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        For each conflict pair in a column, call LLM once to decide winner/loser, then remove loser.
        resolver_chain.invoke input:
          {"table_name","column_name","prop_a_name","prop_b_name"}
        Expected JSON:
          {"winner": "...", "loser": "..."} (names must be exactly one of the pair)
        """
        logs: List[Dict[str, Any]] = []

        for col_res in filtered_results:
            conflicts = col_res.get("inverse_conflicts", []) or []
            if not conflicts:
                continue

            prop_list = list(col_res.get("property", []) or [])
            present = {u for u, _ in prop_list if isinstance(u, str)}

            for pair in conflicts:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                a_name, b_name = pair[0], pair[1]
                if not (isinstance(a_name, str) and isinstance(b_name, str)):
                    continue
                if a_name not in present or b_name not in present:
                    continue

                resp = resolver_chain.invoke(
                    {
                        "table_name": table_name,
                        "column_name": col_res.get("column_name", ""),
                        "prop_a_name": a_name,
                        "prop_b_name": b_name,
                    }
                )
                raw = getattr(resp, "content", "")

                winner = loser = None
                ok = False
                err = ""

                try:
                    obj = json.loads(extract_json_from_response(raw))
                    winner = obj.get("winner")
                    loser = obj.get("loser")
                    if winner in (a_name, b_name) and loser in (a_name, b_name) and winner != loser:
                        ok = True
                    else:
                        err = "invalid_winner_loser"
                except Exception as e:
                    err = f"json_parse_error: {e}"

                logs.append(
                    {
                        "table_name": table_name,
                        "column_name": col_res.get("column_name", ""),
                        "a_name": a_name,
                        "b_name": b_name,
                        "raw": raw,
                        "winner": winner,
                        "loser": loser,
                        "ok": ok,
                        "error": err,
                    }
                )

                if ok and loser is not None:
                    prop_list = [[u, s] for (u, s) in prop_list if u != loser]
                    present = {u for u, _ in prop_list if isinstance(u, str)}

            col_res["property"] = prop_list
            col_res["inverse_conflicts_resolved"] = True

        return filtered_results, logs

    def fix_after_process_llm_output(
        self,
        filtered_results: List[Dict[str, Any]],
        table_name: str,
        resolver_chain,
        *,
        extract_json_from_response: Callable[[str], str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convenience one-shot:
          - attach conflicts
          - resolve conflicts
        Requires self.inverse_map already prepared (call prepare_inverse_conflicts() once before using).
        """
        self.attach_inverse_conflicts(filtered_results)
        return self.resolve_inverse_conflicts(
            filtered_results=filtered_results,
            table_name=table_name,
            resolver_chain=resolver_chain,
            extract_json_from_response=extract_json_from_response,
        )