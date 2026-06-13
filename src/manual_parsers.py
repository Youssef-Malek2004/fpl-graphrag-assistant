# manual_parsers.py
from typing import Any, Dict, List
from neo4j.graph import Node, Relationship, Path
from neo4j import Record


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Node):
        return {
            "type": "node",
            "labels": list(value.labels),
            "properties": {k: _to_jsonable(v) for k, v in dict(value).items()},
        }

    if isinstance(value, Relationship):
        return {
            "type": "relationship",
            "rel_type": value.type,
            "properties": {k: _to_jsonable(v) for k, v in dict(value).items()},
        }

    if isinstance(value, Path):
        return {
            "type": "path",
            "nodes": [_to_jsonable(n) for n in value.nodes],
            "relationships": [_to_jsonable(r) for r in value.relationships],
        }

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (int, float, bool, str)) or value is None:
        return value

    return str(value)


def parse_results_for_llm(intent: str, entities: Dict[str, Any], records: List[Record]) -> Dict[str, Any]:
    """
    Because ALL cypher queries return `item AS item`, results are stable:
      record.data() -> {"item": {...}}
    """
    out_items: List[Dict[str, Any]] = []
    for r in records:
        data = r.data()
        item = data.get("item")
        out_items.append(_to_jsonable(item))
    return {"intent": intent, "entities": entities, "results": out_items}
