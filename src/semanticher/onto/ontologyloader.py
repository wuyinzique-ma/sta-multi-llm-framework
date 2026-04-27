from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import owlready2


class OntologyLoader:
    """
    Final version (token optimized):
      - LLM sees only:
          {"classes":[name,...],
           "properties":[name,...]}
      - Local files:
          - ambiguous_mapping.json (stable disambiguated names for collisions)
          - equivalent_mapping.json (uri_to_canonical, canonical_to_same_as), read-first
      - No URI/labels/comments are exposed to LLM.

    New addition:
      - name_uri_mapping.json
        This is a "final" mapping for your DAG / tools:
          - ambiguous items use disambiguated names (e.g., Building1/Building2)
          - equivalent items are merged (canonical uri + sameAs uris are grouped together)
        File location: same folder as ambiguous_mapping.json / equivalent_mapping.json
    """

    # -------------------------- init --------------------------

    def __init__(
        self,
        rdf_path: str,
        *,
        only_local: bool = False,
        mapping_path: Optional[str] = None,  # points to .../ambiguous_mapping.json
    ):
        self.rdf_path = rdf_path
        self.only_local = only_local
        self.mapping_path = mapping_path

        self.ambiguous_mapping_path: Optional[Path] = None
        self.equivalent_mapping_path: Optional[Path] = None
        if mapping_path:
            mp = Path(mapping_path)
            self.ambiguous_mapping_path = mp
            self.equivalent_mapping_path = mp.parent / "equivalent_mapping.json"

        self._stats: Dict[str, int] = {}

    # -------------------------- public API --------------------------

    def build(self) -> Dict[str, Any]:
        raw = self._load_full()  # includes equivalent merge + equivalent mapping read/write if enabled

        classes = self._dedup_by_uri(raw["classes"])
        op = self._dedup_by_uri(raw["object_properties"])
        dp = self._dedup_by_uri(raw["data_properties"])
        all_props = op + dp

        # Split by name_key collisions (for disambiguation via mapping)
        u_cls, a_cls_groups = self._split_by_name_key(classes)
        u_prop, a_prop_groups = self._split_by_name_key(all_props)

        # ambiguous mapping: read-first; generate/save if missing
        mapping_data: Optional[Dict[str, Any]] = None
        if self.mapping_path:
            assert self.ambiguous_mapping_path is not None
            if os.path.exists(str(self.ambiguous_mapping_path)):
                mapping_data = self._load_mapping_file(str(self.ambiguous_mapping_path))
                mapping_data["_path_hint"] = str(self.ambiguous_mapping_path)
            else:
                onto = owlready2.get_ontology(self.rdf_path).load()
                mapping_data = self._generate_mapping_data(a_cls_groups, a_prop_groups, onto=onto)
                mapping_data["_path_hint"] = str(self.ambiguous_mapping_path)
                self._save_mapping_file(str(self.ambiguous_mapping_path), mapping_data)

        # Format for LLM:
        # - unique: name_key only
        # - ambiguous: mapping_name only (requires mapping_path)
        llm_unique_classes = self._format_unique_for_llm(u_cls)
        llm_unique_props = self._format_unique_for_llm(u_prop)

        llm_amb_classes = self._format_ambiguous_for_llm(a_cls_groups, mapping_data, kind="classes")
        llm_amb_props = self._format_ambiguous_for_llm(a_prop_groups, mapping_data, kind="properties")

        ontology_json: Dict[str, Any] = {
            "classes": llm_unique_classes + llm_amb_classes,
            "properties": llm_unique_props + llm_amb_props,
        }

        # -------- NEW: write name_uri_mapping.json --------
        # We do it here because:
        #   - classes/props are already canonicalized by equivalent mapping in _load_full()
        #   - ambiguous names (Building1/Building2) are already available in mapping_data
        #
        # Output format:
        #   {
        #     "classes":   { name: [uri, uri, ...] },
        #     "properties":{ name: [uri, uri, ...] }
        #   }
        #
        # Note:
        #   - ambiguous items use mapping_data's "name"
        #   - unique items use name_key (same as LLM sees)
        #   - equivalent merging is done by expanding canonical uri with canonical_to_same_as list
        self._write_name_uri_mapping_json(
            u_cls=u_cls,
            a_cls_groups=a_cls_groups,
            u_prop=u_prop,
            a_prop_groups=a_prop_groups,
            mapping_data=mapping_data,
        )

        self._stats = {
            "classes_total": len(classes),
            "classes_unique": len(llm_unique_classes),
            "classes_ambiguous": len(llm_amb_classes),
            "ambiguous_class_groups": len(a_cls_groups),
            "properties_total": len(all_props),
            "properties_unique": len(llm_unique_props),
            "properties_ambiguous": len(llm_amb_props),
            "ambiguous_property_groups": len(a_prop_groups),
            "mapping_enabled": int(bool(self.mapping_path)),
        }
        return ontology_json

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # -------------------------- robust JSON I/O helpers --------------------------

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

    # -------------------------- mapping I/O (ambiguous) --------------------------

    @staticmethod
    def _load_mapping_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must be a JSON object: {path}")
        if "classes" not in data or "properties" not in data:
            raise ValueError(f"Mapping file missing 'classes'/'properties' keys: {path}")
        if not isinstance(data["classes"], list) or not isinstance(data["properties"], list):
            raise ValueError(f"Mapping file 'classes'/'properties' must be lists: {path}")
        return data

    @staticmethod
    def _save_mapping_file(path: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _generate_mapping_data(
        cls,
        a_cls_groups: Dict[str, List[Dict[str, str]]],
        a_prop_groups: Dict[str, List[Dict[str, str]]],
        onto=None,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "classes":    cls._generate_mapping_entries(a_cls_groups, onto=onto),
            "properties": cls._generate_mapping_entries(a_prop_groups, onto=onto),
        }

    @classmethod
    def _generate_mapping_entries(cls, groups: Dict[str, List[Dict[str, str]]], onto=None) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for base in sorted(groups.keys(), key=lambda s: s.lower()):
            items = sorted(groups[base], key=lambda x: (x.get("uri") or ""))
            for idx, it in enumerate(items, start=1):
                uri = (it.get("uri") or "").strip()
                if not uri:
                    continue

                comment = ""
                parents = []

                if onto is not None:
                    try:
                        entity = onto.search_one(iri=uri)
                        if entity is not None:
                            # comment
                            raw_comment = getattr(entity, "comment", None)
                            if raw_comment:
                                for v in raw_comment:
                                    if isinstance(v, str) and v.strip():
                                        comment = v.strip()
                                        break
                            # parents
                            for parent in getattr(entity, "is_a", []):
                                parent_iri = getattr(parent, "iri", None)
                                if parent_iri:
                                    parent_name = cls._uri_local_name(str(parent_iri))
                                    if parent_name:
                                        parents.append(parent_name)
                    except Exception:
                        pass

                entry = {
                    "name":    f"{base}{idx}",
                    "label":   base,
                    "uri":     uri,
                    "comment": comment,
                    "parents": parents,
                }
                entries.append(entry)
        return entries

    # -------------------------- equivalent mapping I/O (read-first) --------------------------

    def _load_equivalent_mapping_uri_to_canonical(self) -> Optional[Dict[str, str]]:
        data = self._safe_read_json(self.equivalent_mapping_path)
        if not data:
            return None
        utc = data.get("uri_to_canonical")
        if not isinstance(utc, dict):
            return None
        out: Dict[str, str] = {}
        for k, v in utc.items():
            if isinstance(k, str) and isinstance(v, str):
                ks, vs = k.strip(), v.strip()
                if ks and vs and ks != vs:
                    out[ks] = vs
        return out

    def _save_equivalent_mapping(self, rep_map: Dict[str, str]) -> None:
        """
        rep_map: uri -> canonical_uri (may include self-mapping)
        Save:
          - uri_to_canonical: only non-self mappings
          - canonical_to_same_as: reverse index (debug)
        """
        uri_to_canonical: Dict[str, str] = {}
        canonical_to_same_as: Dict[str, List[str]] = {}

        for uri, canon in rep_map.items():
            uri = (uri or "").strip()
            canon = (canon or "").strip()
            if not uri or not canon or uri == canon:
                continue
            uri_to_canonical[uri] = canon
            canonical_to_same_as.setdefault(canon, []).append(uri)

        uri_to_canonical = dict(sorted(uri_to_canonical.items(), key=lambda x: x[0]))
        for canon in list(canonical_to_same_as.keys()):
            canonical_to_same_as[canon] = sorted(set(canonical_to_same_as[canon]))
        canonical_to_same_as = dict(sorted(canonical_to_same_as.items(), key=lambda x: x[0]))

        payload = {
            "uri_to_canonical": uri_to_canonical,
            "canonical_to_same_as": canonical_to_same_as,
        }
        assert self.equivalent_mapping_path is not None
        self._atomic_write_json(self.equivalent_mapping_path, payload)

    def _load_equivalent_canonical_to_same_as(self) -> Dict[str, List[str]]:
        """
        Read canonical_to_same_as from equivalent_mapping.json.
        This is used to expand a canonical uri back to a full uri list.
        """
        data = self._safe_read_json(self.equivalent_mapping_path)
        if not data:
            return {}
        c2s = data.get("canonical_to_same_as")
        if not isinstance(c2s, dict):
            return {}

        out: Dict[str, List[str]] = {}
        for canon, lst in c2s.items():
            if not isinstance(canon, str):
                continue
            canon = canon.strip()
            if not canon:
                continue
            if isinstance(lst, list):
                vals = [x.strip() for x in lst if isinstance(x, str) and x.strip()]
            else:
                vals = []
            out[canon] = sorted(set(vals))
        return out

    # -------------------------- Owlready2 helpers --------------------------

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

    # -------------------------- equivalent merge --------------------------

    class _UnionFind:
        def __init__(self):
            self.parent: Dict[str, str] = {}

        def find(self, x: str) -> str:
            p = self.parent.get(x, x)
            if p != x:
                p = self.find(p)
                self.parent[x] = p
            return p

        def union(self, a: str, b: str) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    @staticmethod
    def _safe_iri(x) -> Optional[str]:
        iri = getattr(x, "iri", None)
        if iri:
            s = str(iri).strip()
            return s if s else None
        s = str(x).strip()
        return s if (s.startswith("http://") or s.startswith("https://")) else None

    @classmethod
    def _extract_equivalent_iris(cls, entity) -> List[str]:
        out: List[str] = []
        eq_list = getattr(entity, "equivalent_to", None) or []
        for y in eq_list:
            iri = cls._safe_iri(y)
            if iri:
                out.append(iri)
        return out

    @classmethod
    def _build_equivalence_rep_map(cls, entities: List[Any]) -> Dict[str, str]:
        """
        Build uri -> canonical_uri for entities (deterministic).
        Canonical chosen as lexicographically smallest member among included entities.
        """
        uf = cls._UnionFind()
        included: List[str] = []
        for e in entities:
            iri = cls._safe_iri(e)
            if iri:
                included.append(iri)
        included_set = set(included)

        for e in entities:
            e_iri = cls._safe_iri(e)
            if not e_iri:
                continue
            for other_iri in cls._extract_equivalent_iris(e):
                uf.union(e_iri, other_iri)

        groups: Dict[str, List[str]] = {}
        for iri in included_set:
            root = uf.find(iri)
            groups.setdefault(root, []).append(iri)

        rep_map: Dict[str, str] = {}
        for members in groups.values():
            canon = sorted(set(members))[0]
            for m in set(members):
                rep_map[m] = canon
        return rep_map

    @classmethod
    def _merge_items_by_rep_map(cls, items: List[Dict[str, str]], rep_map: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Canonicalize uri + merge label deterministically.
        Output items keep uri/label internally.
        """
        buckets: Dict[str, List[Dict[str, str]]] = {}
        for it in items:
            uri = (it.get("uri") or "").strip()
            if not uri:
                continue
            canon = rep_map.get(uri, uri)
            buckets.setdefault(canon, []).append(it)

        merged: List[Dict[str, str]] = []
        for canon, group in buckets.items():
            group_sorted = sorted(group, key=lambda x: (x.get("uri") or ""))
            label = ""
            for g in group_sorted:
                if not label and (g.get("label") or "").strip():
                    label = g["label"].strip()
                    break
            merged.append({"uri": canon, "label": label})

        merged.sort(key=lambda x: x["uri"])
        return merged

    # -------------------------- loading --------------------------

    def _load_full(self) -> Dict[str, List[Dict[str, str]]]:
        onto = owlready2.get_ontology(self.rdf_path).load()

        def is_local(x) -> bool:
            return (not self.only_local) or (getattr(x, "namespace", None) == onto)

        class_ents = [c for c in onto.classes() if is_local(c)]
        op_ents = [p for p in onto.object_properties() if is_local(p)]
        dp_ents = [p for p in onto.data_properties() if is_local(p)]

        # Equivalent mapping: read-first; generate only if missing (when mapping_path provided)
        rep_map: Dict[str, str] = {}
        if self.mapping_path:
            assert self.equivalent_mapping_path is not None
            loaded = self._load_equivalent_mapping_uri_to_canonical()
            if loaded is not None:
                rep_map = dict(loaded)  # non-self only; identity fallback in use
            else:
                rep_map = self._build_equivalence_rep_map(class_ents + op_ents + dp_ents)
                self._save_equivalent_mapping(rep_map)
        else:
            rep_map = self._build_equivalence_rep_map(class_ents + op_ents + dp_ents)

        def pack(entity) -> Dict[str, str]:
            uri = str(getattr(entity, "iri", None) or str(entity))
            label = (self._get_label(entity) or getattr(entity, "name", "") or self._uri_local_name(uri)).strip()
            return {"uri": uri, "label": label}

        classes_raw = [pack(c) for c in class_ents]
        op_raw = [pack(p) for p in op_ents]
        dp_raw = [pack(p) for p in dp_ents]

        classes = self._merge_items_by_rep_map(classes_raw, rep_map)
        object_properties = self._merge_items_by_rep_map(op_raw, rep_map)
        data_properties = self._merge_items_by_rep_map(dp_raw, rep_map)

        return {
            "classes": classes,
            "object_properties": object_properties,
            "data_properties": data_properties,
        }

    # -------------------------- normalization & grouping --------------------------

    @staticmethod
    def _uri_local_name(uri: str) -> str:
        if not isinstance(uri, str) or not uri:
            return ""
        if "#" in uri:
            return uri.rsplit("#", 1)[-1].strip()
        if "/" in uri:
            return uri.rsplit("/", 1)[-1].strip()
        return uri.strip()

    @staticmethod
    def _dedup_by_uri(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out: List[Dict[str, str]] = []
        for x in items:
            u = x.get("uri")
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(x)
        return out

    @classmethod
    def _name_key(cls, it: Dict[str, str]) -> str:
        # Policy: prefer label; fallback to uri local name
        label = (it.get("label") or "").strip()
        if label:
            return label
        return cls._uri_local_name(it.get("uri", ""))

    @classmethod
    def _split_by_name_key(
        cls, items: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        counts: Dict[str, int] = {}
        keys: List[str] = []

        for it in items:
            k = cls._name_key(it)
            keys.append(k)
            counts[k] = counts.get(k, 0) + 1

        unique_items: List[Dict[str, str]] = []
        ambiguous_groups: Dict[str, List[Dict[str, str]]] = {}

        for it, k in zip(items, keys):
            if counts.get(k, 0) == 1:
                unique_items.append(it)
            else:
                ambiguous_groups.setdefault(k, []).append(it)

        return unique_items, ambiguous_groups

    # -------------------------- formatting for LLM --------------------------

    @classmethod
    def _format_unique_for_llm(cls, items: List[Dict[str, str]]) -> List[str]:
        out: List[str] = []
        for it in items:
            out.append(cls._name_key(it))
        return out

    @classmethod
    def _format_ambiguous_for_llm(
        cls,
        groups: Dict[str, List[Dict[str, str]]],
        mapping_data: Optional[Dict[str, Any]],
        *,
        kind: str,  # "classes" or "properties"
    ) -> List[str]:
        if not groups:
            return []

        if mapping_data is None:
            raise ValueError(
                "mapping_path is required for ambiguous handling. "
                "Provide mapping_path or ensure ontology has no ambiguous items."
            )

        entries = mapping_data.get(kind, [])
        if not isinstance(entries, list):
            raise ValueError(f"Mapping '{kind}' must be a list in mapping file.")

        # uri -> mapping record
        uri_to_rec: Dict[str, Dict[str, str]] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            uri = (e.get("uri") or "").strip()
            name = (e.get("name") or "").strip()
            if uri and name:
                uri_to_rec[uri] = {"name": name, "uri": uri}

        # coverage check
        missing: List[Tuple[str, str]] = []
        for base in sorted(groups.keys(), key=lambda s: s.lower()):
            for it in groups[base]:
                uri = (it.get("uri") or "").strip()
                if uri and uri not in uri_to_rec:
                    missing.append((base, uri))

        if missing:
            preview = "\n".join([f"  - label='{b}', uri='{u}'" for b, u in missing[:20]])
            more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
            path_hint = mapping_data.get("_path_hint", "(unknown)")
            raise ValueError(
                f"Mapping file does not cover all ambiguous {kind} in the current ontology.\n"
                f"Mapping file: {path_hint}\n"
                f"Missing entries:\n{preview}{more}\n\n"
                f"Fix: delete the mapping file to regenerate it from current ontology, "
                f"or add the missing entries manually."
            )

        # build LLM list (strings only)
        llm_items: List[str] = []
        for base in sorted(groups.keys(), key=lambda s: s.lower()):
            items = sorted(groups[base], key=lambda x: (x.get("uri") or ""))
            for it in items:
                uri = (it.get("uri") or "").strip()
                llm_items.append(uri_to_rec[uri]["name"])

        return llm_items

    # -------------------------- NEW: write name_uri_mapping.json --------------------------

    @staticmethod
    def _uri_to_disamb_name(mapping_data: Optional[Dict[str, Any]], kind: str) -> Dict[str, str]:
        if not mapping_data:
            return {}
        entries = mapping_data.get(kind, [])
        if not isinstance(entries, list):
            return {}

        out: Dict[str, str] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            uri = (e.get("uri") or "").strip()
            name = (e.get("name") or "").strip()
            if uri and name:
                out[uri] = name
        return out

    def _write_name_uri_mapping_json(
        self,
        *,
        u_cls: List[Dict[str, str]],
        a_cls_groups: Dict[str, List[Dict[str, str]]],
        u_prop: List[Dict[str, str]],
        a_prop_groups: Dict[str, List[Dict[str, str]]],
        mapping_data: Optional[Dict[str, Any]],
    ) -> None:
        """
        Write name_uri_mapping.json in the same folder as ambiguous/equivalent mapping files.

        Rules:
          - unique items use name_key
          - ambiguous items use disambiguated name from mapping_data
          - equivalent items are merged by expanding canonical uri to [canonical + sameAs...]
        """
        # decide output folder
        if self.mapping_path:
            base_dir = Path(self.mapping_path).parent
        else:
            base_dir = Path(self.rdf_path).parent

        out_path = base_dir / "name_uri_mapping.json"

        # load equivalent expansion (only meaningful when mapping_path is enabled)
        canonical_to_same_as: Dict[str, List[str]] = {}
        if self.mapping_path:
            canonical_to_same_as = self._load_equivalent_canonical_to_same_as()

        def expand_uris(canon_uri: str) -> List[str]:
            canon_uri = (canon_uri or "").strip()
            if not canon_uri:
                return []
            extra = canonical_to_same_as.get(canon_uri, [])
            return sorted(set([canon_uri] + list(extra)))

        # ambiguous uri -> disambiguated name
        cls_uri_to_name = self._uri_to_disamb_name(mapping_data, "classes")
        prop_uri_to_name = self._uri_to_disamb_name(mapping_data, "properties")

        final_map: Dict[str, Dict[str, List[str]]] = {"classes": {}, "properties": {}}

        # classes (unique)
        for it in u_cls:
            name = self._name_key(it)
            uris = expand_uris(it.get("uri", ""))
            if name and uris:
                final_map["classes"][name] = uris

        # classes (ambiguous)
        for base in sorted(a_cls_groups.keys(), key=lambda s: s.lower()):
            items = sorted(a_cls_groups[base], key=lambda x: (x.get("uri") or ""))
            for it in items:
                uri = (it.get("uri") or "").strip()
                name = cls_uri_to_name.get(uri, "")
                uris = expand_uris(uri)
                if name and uris:
                    final_map["classes"][name] = uris

        # properties (unique)
        for it in u_prop:
            name = self._name_key(it)
            uris = expand_uris(it.get("uri", ""))
            if name and uris:
                final_map["properties"][name] = uris

        # properties (ambiguous)
        for base in sorted(a_prop_groups.keys(), key=lambda s: s.lower()):
            items = sorted(a_prop_groups[base], key=lambda x: (x.get("uri") or ""))
            for it in items:
                uri = (it.get("uri") or "").strip()
                name = prop_uri_to_name.get(uri, "")
                uris = expand_uris(uri)
                if name and uris:
                    final_map["properties"][name] = uris

        self._atomic_write_json(out_path, final_map)