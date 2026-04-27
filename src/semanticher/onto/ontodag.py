from pathlib import Path
import json
from collections import defaultdict

import owlready2
from owlready2 import Thing

from semanticher.onto.ontoclass import OntologyClass

_HERE = Path(__file__).resolve().parent
_DEFAULT_RDF_PATH         = (_HERE / ".." / ".." / ".." / "data" / "ontology" / "BED.owl").resolve()
_DEFAULT_MAPPING_PATH     = (_HERE / ".." / ".." / ".." / "data" / "ontology" / "name_uri_mapping.json").resolve()


class OntologyDAG:
    def __init__(self, rdf_file_path=_DEFAULT_RDF_PATH, name_uri_mapping_path=_DEFAULT_MAPPING_PATH):
        """
        Represents the ontology as a Directed Acyclic Graph (DAG).

        Attributes:
            nodes (dict): A dictionary to store nodes with URI as keys and OntologyClass instances as values.
            edges (defaultdict): A dictionary storing parent → [children] relationships.
            edges_subclassof (defaultdict): A dictionary storing child → [parents] relationships.
            root (str): The URI of the root class in the ontology.

        Mapping attributes:
            name_to_uris (dict): A dictionary storing name to URIs mapping loaded from name_uri_mapping.json.
                                 It is mainly used for "name to uris" lookup.
        """
        self.rdf_file_path = rdf_file_path

        self.nodes           = {}                  # URI → OntologyClass
        self.edges           = defaultdict(list)   # parent URI → [children URIs]
        self.edges_subclassof = defaultdict(list)  # child URI  → [parent URIs]
        self.root            = None

        self.name_uri_mapping_path = name_uri_mapping_path
        self._mapping_loaded       = False

        self._name_to_uris      = {}   # class name → [URIs]
        self._uri_to_name_index = {}   # URI → class name (for display/grouping)

    # -------------------------- mapping --------------------------

    def _load_name_uri_mapping(self):
        if self._mapping_loaded:
            return
        self._mapping_loaded = True

        path = self.name_uri_mapping_path
        if not path or not Path(path).exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        classes = data.get("classes", {})
        if not isinstance(classes, dict):
            return

        name_to_uris = {}
        uri_to_name  = {}

        for name, uris in classes.items():
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()

            if isinstance(uris, str):
                uris = [uris]
            if not isinstance(uris, list):
                continue

            clean = sorted(set(u.strip() for u in uris if isinstance(u, str) and u.strip()))
            if not clean:
                continue

            name_to_uris[name] = clean
            for u in clean:
                uri_to_name[u] = name

        self._name_to_uris      = name_to_uris
        self._uri_to_name_index = uri_to_name

    @property
    def name_to_uris(self):
        self._load_name_uri_mapping()
        return self._name_to_uris

    # -------------------------- naming helpers --------------------------

    def _uri_local_name(self, uri):
        if not isinstance(uri, str) or not uri:
            return ""
        if "#" in uri:
            return uri.rsplit("#", 1)[-1].strip()
        if "/" in uri:
            return uri.rsplit("/", 1)[-1].strip()
        return uri.strip()

    def name_of_uri(self, uri, fallback=None):
        """
        Return a display name for a URI.

        Priority:
            1) name_uri_mapping.json (uri → name)
            2) fallback
            3) local part of URI
        """
        self._load_name_uri_mapping()

        name = self._uri_to_name_index.get(uri)
        if name:
            return name

        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()

        node = self.nodes.get(uri)
        if node is not None:
            n = getattr(node, "name", None)
            if isinstance(n, str) and n.strip():
                return n.strip()

        return self._uri_local_name(uri) or uri

    # -------------------------- per-level grouping (for BFS) --------------------------

    def group_by_name(self, candidate_uris):
        """
        Group a list of candidate URIs by their display name.

        Recommended for BFS:
            - LLM receives only unique names (keys of this dict).
            - When LLM selects a name, only URIs present in the current candidate set
              are expanded (prevents jumping to other branches).
        """
        self._load_name_uri_mapping()

        groups = defaultdict(list)
        for u in candidate_uris:
            if isinstance(u, str) and u:
                groups[self.name_of_uri(u)].append(u)

        return dict(groups)

    def uris_for_name(self, name, *, candidate_uris=None):
        """
        Look up URIs for a given name from name_uri_mapping.json.

        If candidate_uris is provided, return only URIs that also appear in candidate_uris.
        This prevents expanding to URIs outside the current BFS level.
        """
        self._load_name_uri_mapping()

        if not isinstance(name, str) or not name.strip():
            return []

        uris = self._name_to_uris.get(name.strip(), [])
        if not uris:
            return []

        if candidate_uris is None:
            return list(uris)

        s = {u for u in candidate_uris if isinstance(u, str) and u}
        return [u for u in uris if u in s]

    # -------------------------- build DAG --------------------------

    def build_dag(self, rdf_file_path=None):
        """
        Builds the ontology DAG from an RDF file.

        Args:
            rdf_file_path (str | Path): Path to the RDF/OWL file.

        After building:
            edges[parent]           → [children]   (parent → children, used for BFS traversal)
            edges_subclassof[child] → [parents]    (child  → parents,  subclassOf semantics)
        """
        if rdf_file_path is not None:
            self.rdf_file_path = rdf_file_path

        onto = owlready2.get_ontology(str(self.rdf_file_path)).load()

        self.nodes            = {}
        self.edges            = defaultdict(list)
        self.edges_subclassof = defaultdict(list)
        self.root             = None

        for cls in onto.classes():
            uri = cls.iri
            self.nodes[uri] = OntologyClass(uri, name=cls.name, label=cls.label, comment=cls.comment)

            for child in cls.subclasses():
                if child.iri == uri:
                    continue
                self.edges[uri].append(child.iri)              # parent → children
                self.edges_subclassof[child.iri].append(uri)   # child  → parents

        self.root = Thing.iri

    def __repr__(self):
        return f"OntologyDAG(nodes={list(self.nodes.keys())}, edges={dict(self.edges)})"


if __name__ == "__main__":
    dag = OntologyDAG()
    dag.build_dag()

    n_level1, n_level2 = 0, 0
    for o in dag.edges.get(dag.root, []):
        n_level1 += 1
        print(dag.name_of_uri(o))
        for c in dag.edges.get(o, []):
            n_level2 += 1
            print("\t", dag.name_of_uri(c))
    print(f"Number of level 1: {n_level1}")
    print(f"Number of level 2: {n_level2}")