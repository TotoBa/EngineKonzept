"""Schema definitions for dataset examples and Phase-5 artifact loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

SUPPORTED_RESULTS = {"1-0", "0-1", "1/2-1/2"}
SUPPORTED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class SplitRatios:
    """Deterministic split ratios for dataset generation."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        ratios = (self.train, self.validation, self.test)
        if any(ratio < 0.0 for ratio in ratios):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")


@dataclass(frozen=True)
class RawPositionRecord:
    """Minimal raw position record before exact-rule enrichment."""

    sample_id: str
    fen: str
    source: str
    selected_move_uci: str | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.fen:
            raise ValueError("fen must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.result is not None and self.result not in SUPPORTED_RESULTS:
            raise ValueError(f"unsupported result label: {self.result}")

    def to_oracle_input(self) -> dict[str, Any]:
        """Convert the record to the Rust oracle request schema."""
        return {
            "fen": self.fen,
            "selected_move_uci": self.selected_move_uci,
        }


@dataclass(frozen=True)
class WdlTarget:
    """One-hot WDL target relative to the side to move."""

    win: int
    draw: int
    loss: int

    def __post_init__(self) -> None:
        if (self.win, self.draw, self.loss).count(1) != 1:
            raise ValueError("wdl target must be one-hot")
        if any(value not in {0, 1} for value in (self.win, self.draw, self.loss)):
            raise ValueError("wdl target values must be 0 or 1")

    def to_dict(self) -> dict[str, int]:
        """Return the JSON representation."""
        return {"win": self.win, "draw": self.draw, "loss": self.loss}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WdlTarget":
        """Construct a WDL target from its JSON representation."""
        return cls(
            win=int(payload["win"]),
            draw=int(payload["draw"]),
            loss=int(payload["loss"]),
        )


@dataclass(frozen=True)
class PositionEncoding:
    """JSON-friendly encoded position matrices."""

    piece_tokens: list[list[int]]
    square_tokens: list[list[int]]
    rule_token: list[int]

    def __post_init__(self) -> None:
        for token in self.piece_tokens:
            if len(token) != 3:
                raise ValueError("piece tokens must have width 3")
        if len(self.square_tokens) != 64:
            raise ValueError("square token matrix must have 64 rows")
        for token in self.square_tokens:
            if len(token) != 2:
                raise ValueError("square tokens must have width 2")
        if len(self.rule_token) != 6:
            raise ValueError("rule token must have width 6")

    @classmethod
    def from_oracle_dict(cls, payload: dict[str, Any]) -> "PositionEncoding":
        """Construct the encoding from the Rust oracle JSON payload."""
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PositionEncoding":
        """Construct the encoding from JSON."""
        return cls(
            piece_tokens=[[int(value) for value in token] for token in payload["piece_tokens"]],
            square_tokens=[[int(value) for value in token] for token in payload["square_tokens"]],
            rule_token=[int(value) for value in payload["rule_token"]],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation."""
        return {
            "piece_tokens": self.piece_tokens,
            "square_tokens": self.square_tokens,
            "rule_token": self.rule_token,
        }


@dataclass(frozen=True)
class TacticalAnnotations:
    """Exact and conservative annotations emitted by the dataset oracle."""

    in_check: bool
    is_checkmate: bool
    is_stalemate: bool
    has_legal_en_passant: bool
    has_legal_castle: bool
    has_legal_promotion: bool
    is_low_material_endgame: bool
    legal_move_count: int
    piece_count: int
    selected_move_is_capture: bool | None
    selected_move_is_promotion: bool | None
    selected_move_is_castle: bool | None
    selected_move_is_en_passant: bool | None
    selected_move_gives_check: bool | None

    @classmethod
    def from_oracle_dict(cls, payload: dict[str, Any]) -> "TacticalAnnotations":
        """Construct annotations from the Rust oracle JSON payload."""
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TacticalAnnotations":
        """Construct annotations from JSON."""
        return cls(
            in_check=bool(payload["in_check"]),
            is_checkmate=bool(payload["is_checkmate"]),
            is_stalemate=bool(payload["is_stalemate"]),
            has_legal_en_passant=bool(payload["has_legal_en_passant"]),
            has_legal_castle=bool(payload["has_legal_castle"]),
            has_legal_promotion=bool(payload["has_legal_promotion"]),
            is_low_material_endgame=bool(payload["is_low_material_endgame"]),
            legal_move_count=int(payload["legal_move_count"]),
            piece_count=int(payload["piece_count"]),
            selected_move_is_capture=_optional_bool(payload["selected_move_is_capture"]),
            selected_move_is_promotion=_optional_bool(payload["selected_move_is_promotion"]),
            selected_move_is_castle=_optional_bool(payload["selected_move_is_castle"]),
            selected_move_is_en_passant=_optional_bool(payload["selected_move_is_en_passant"]),
            selected_move_gives_check=_optional_bool(payload["selected_move_gives_check"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation."""
        return {
            "in_check": self.in_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "has_legal_en_passant": self.has_legal_en_passant,
            "has_legal_castle": self.has_legal_castle,
            "has_legal_promotion": self.has_legal_promotion,
            "is_low_material_endgame": self.is_low_material_endgame,
            "legal_move_count": self.legal_move_count,
            "piece_count": self.piece_count,
            "selected_move_is_capture": self.selected_move_is_capture,
            "selected_move_is_promotion": self.selected_move_is_promotion,
            "selected_move_is_castle": self.selected_move_is_castle,
            "selected_move_is_en_passant": self.selected_move_is_en_passant,
            "selected_move_gives_check": self.selected_move_gives_check,
        }


@dataclass(frozen=True)
class DatasetExample:
    """Fully enriched training example emitted by the builder."""

    sample_id: str
    split: str
    source: str
    fen: str
    side_to_move: str
    selected_move_uci: str | None
    selected_action_encoding: list[int] | None
    next_fen: str | None
    legal_moves: list[str]
    legal_action_encodings: list[list[int]]
    position_encoding: PositionEncoding
    wdl_target: WdlTarget | None
    annotations: TacticalAnnotations
    result: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split name: {self.split}")
        if self.side_to_move not in {"w", "b"}:
            raise ValueError("side_to_move must be 'w' or 'b'")
        if self.selected_action_encoding is not None and len(self.selected_action_encoding) != 3:
            raise ValueError("selected_action_encoding must contain 3 indices")
        for action in self.legal_action_encodings:
            if len(action) != 3:
                raise ValueError("legal_action_encodings entries must contain 3 indices")
        if self.selected_move_uci is None:
            if self.selected_action_encoding is not None or self.next_fen is not None:
                raise ValueError("selected_action_encoding and next_fen require selected_move_uci")
        else:
            if self.selected_action_encoding is None or self.next_fen is None:
                raise ValueError("selected_move_uci requires selected_action_encoding and next_fen")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "selected_move_uci": self.selected_move_uci,
            "selected_action_encoding": self.selected_action_encoding,
            "next_fen": self.next_fen,
            "legal_moves": self.legal_moves,
            "legal_action_encodings": self.legal_action_encodings,
            "position_encoding": self.position_encoding.to_dict(),
            "wdl_target": None if self.wdl_target is None else self.wdl_target.to_dict(),
            "annotations": self.annotations.to_dict(),
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetExample":
        """Construct an example from the serialized JSON form."""
        return cls(
            sample_id=str(payload["sample_id"]),
            split=str(payload["split"]),
            source=str(payload["source"]),
            fen=str(payload["fen"]),
            side_to_move=str(payload["side_to_move"]),
            selected_move_uci=_optional_string(payload.get("selected_move_uci")),
            selected_action_encoding=_optional_int_list(payload.get("selected_action_encoding")),
            next_fen=_optional_string(payload.get("next_fen")),
            legal_moves=[str(move) for move in payload["legal_moves"]],
            legal_action_encodings=[
                [int(value) for value in action] for action in payload["legal_action_encodings"]
            ],
            position_encoding=PositionEncoding.from_dict(dict(payload["position_encoding"])),
            wdl_target=_optional_wdl_target(payload.get("wdl_target")),
            annotations=TacticalAnnotations.from_dict(dict(payload["annotations"])),
            result=_optional_string(payload.get("result")),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "DatasetExample":
        """Parse an example from a JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: dataset example must be a JSON object")
        return cls.from_dict(payload)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_wdl_target(value: Any) -> WdlTarget | None:
    if value is None:
        return None
    return WdlTarget.from_dict(dict(value))
