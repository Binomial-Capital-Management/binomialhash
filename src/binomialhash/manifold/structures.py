"""Core manifold data structures for the BinomialHash package."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ManifoldAxis:
    """One inferred parameter-space axis in the manifold view."""

    column: str
    values: List[Any]
    ordered: bool
    axis_type: str
    size: int
    wraps: bool = False
    wrap_orientation: int = 0


@dataclass
class GridPoint:
    """A vertex in the discrete manifold grid."""

    index: int
    axis_coords: Tuple
    field_values: Dict[str, float]
    curvature: float = 0.0
    forman_ricci: float = 0.0
    density: int = 0
    neighbors: List[int] = field(default_factory=list)
    morse_type: Dict[str, str] = field(default_factory=dict)


@dataclass
class CriticalPoint:
    """A Morse-like critical point on the grid."""

    vertex_index: int
    coord: Tuple
    morse_type: str
    field: str
    value: float
    persistence: float = 0.0


@dataclass
class ManifoldSurface:
    """Full manifold summary object returned by the builder."""

    axes: List[ManifoldAxis]
    field_columns: List[str]
    grid: Dict[Tuple, GridPoint]
    surface_name: str
    genus: int
    orientable: bool
    euler_characteristic: int
    handles: int
    crosscaps: int
    vertex_count: int
    edge_count: int
    betti_0: int = 0
    betti_1: int = 0
    persistence_pairs: Dict[str, List[CriticalPoint]] = field(default_factory=dict)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    classification_mode: str = "graph_heuristic"
    product_topology_label: str = ""
    surface_confidence_obj: Dict[str, Any] = field(default_factory=dict)
    face_count: int = 0
    face_coverage: float = 0.0
    occupancy: float = 0.0
    face_vertex_count: int = 0
    face_edge_count: int = 0
    boundary_edge_count: int = 0
    boundary_loop_count: int = 0
    face_euler_characteristic: Optional[int] = None
    face_genus_estimate: Optional[float] = None
    face_crosscap_estimate: Optional[int] = None
    is_manifold: bool = False
    nonmanifold_edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a compact manifold summary."""

        return {
            "axes": [
                {
                    "column": a.column,
                    "ordered": a.ordered,
                    "axis_type": a.axis_type,
                    "size": a.size,
                    "wraps": a.wraps,
                    "wrap_orientation": a.wrap_orientation,
                }
                for a in self.axes
            ],
            "fields": self.field_columns,
            "vertices": self.vertex_count,
            "edges": self.edge_count,
            "surface": self.surface_name,
            "genus": self.genus,
            "orientable": self.orientable,
            "euler_characteristic": self.euler_characteristic,
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "handles": self.handles,
            "crosscaps": self.crosscaps,
            "face_complex": {
                "vertices": self.face_vertex_count,
                "edges": self.face_edge_count,
                "faces": self.face_count,
                "boundary_edges": self.boundary_edge_count,
                "boundary_loops": self.boundary_loop_count,
                "face_coverage": self.face_coverage,
                "occupancy": self.occupancy,
                "euler_characteristic": self.face_euler_characteristic,
                "genus_estimate": self.face_genus_estimate,
                "crosscap_estimate": self.face_crosscap_estimate,
                "is_manifold": self.is_manifold,
                "nonmanifold_edges": self.nonmanifold_edges,
            },
            "classification_mode": self.classification_mode,
            "product_topology_label": self.product_topology_label,
            "surface_confidence": self.surface_confidence_obj,
            "interactions": self.interactions,
        }
