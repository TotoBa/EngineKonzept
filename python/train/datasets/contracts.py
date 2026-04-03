"""Versioned symbolic feature contracts shared across proposer, dynamics, and workflows."""

from __future__ import annotations

from dataclasses import dataclass

SYMBOLIC_MAX_LEGAL_CANDIDATES = 256
DEFAULT_CANDIDATE_CONTEXT_VERSION = 1
DEFAULT_GLOBAL_CONTEXT_VERSION = 1

_CANDIDATE_CONTEXT_V1_FEATURE_ORDER = (
    "is_capture",
    "is_promotion",
    "is_castle",
    "is_en_passant",
    "gives_check",
    "from_attacked_by_opponent",
    "to_attacked_by_opponent",
    "from_defended_by_self",
    "to_attacked_by_self",
    "moving_piece_pawn",
    "moving_piece_knight",
    "moving_piece_bishop",
    "moving_piece_rook",
    "moving_piece_queen",
    "moving_piece_king",
    "captured_piece_present",
    "captured_piece_pawn",
    "captured_piece_minor_or_major",
)

_CANDIDATE_CONTEXT_V2_FEATURE_ORDER = (
    "is_capture",
    "is_promotion",
    "is_castle",
    "is_en_passant",
    "gives_check",
    "pre_from_square_attacked_by_opponent",
    "pre_to_square_attacked_by_opponent",
    "pre_from_square_defended_by_self",
    "pre_to_square_attacked_by_self",
    "moving_piece_pawn",
    "moving_piece_knight",
    "moving_piece_bishop",
    "moving_piece_rook",
    "moving_piece_queen",
    "moving_piece_king",
    "captured_piece_present",
    "captured_piece_pawn",
    "captured_piece_knight",
    "captured_piece_bishop",
    "captured_piece_rook",
    "captured_piece_queen",
    "promotion_to_knight",
    "promotion_to_bishop",
    "promotion_to_rook",
    "promotion_to_queen",
    "castle_kingside",
    "castle_queenside",
    "from_file_normalized",
    "from_rank_normalized",
    "to_file_normalized",
    "to_rank_normalized",
    "delta_file_normalized",
    "delta_rank_normalized",
    "abs_delta_file_normalized",
    "abs_delta_rank_normalized",
)

_GLOBAL_CONTEXT_V1_FEATURE_ORDER = (
    "in_check",
    "has_legal_castle",
    "has_legal_en_passant",
    "has_legal_promotion",
    "is_low_material_endgame",
    "legal_move_count_normalized",
    "piece_count_normalized",
    "self_attack_square_ratio",
    "opponent_attack_square_ratio",
)
_TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER = (
    "opponent_in_check_after_move",
    "destination_attacked_after_move",
    "destination_defended_after_move",
    "halfmove_reset",
    "white_kingside_castling_cleared",
    "white_queenside_castling_cleared",
    "black_kingside_castling_cleared",
    "black_queenside_castling_cleared",
    "en_passant_created",
    "en_passant_cleared",
)


@dataclass(frozen=True)
class FeatureContractSpec:
    """Structured description of one symbolic feature contract."""

    contract_name: str
    version: int
    feature_order: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_order)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_name": self.contract_name,
            "version": self.version,
            "feature_dim": self.feature_dim,
            "feature_order": list(self.feature_order),
        }


def candidate_context_spec(version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION) -> FeatureContractSpec:
    """Return the named candidate-context contract."""
    feature_order = _candidate_context_feature_order(version)
    return FeatureContractSpec(
        contract_name="CandidateContext",
        version=version,
        feature_order=feature_order,
    )


def global_context_spec(version: int = DEFAULT_GLOBAL_CONTEXT_VERSION) -> FeatureContractSpec:
    """Return the named global-context contract."""
    feature_order = _global_context_feature_order(version)
    return FeatureContractSpec(
        contract_name="GlobalContext",
        version=version,
        feature_order=feature_order,
    )


def candidate_context_feature_order(
    version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
) -> tuple[str, ...]:
    """Return the feature order for one candidate-context version."""
    return _candidate_context_feature_order(version)


def candidate_context_feature_dim(version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION) -> int:
    """Return the feature width for one candidate-context version."""
    return len(_candidate_context_feature_order(version))


def project_candidate_context_to_v1(
    feature_values: list[float] | tuple[float, ...],
    *,
    version: int,
) -> list[float]:
    """Project a versioned candidate-context row onto the V1 dynamics contract."""
    values = [float(value) for value in feature_values]
    if version == 1:
        expected_width = candidate_context_feature_dim(1)
        if len(values) != expected_width:
            raise ValueError(
                f"CandidateContextV1 row must have width {expected_width}, got {len(values)}"
            )
        return values
    if version != 2:
        raise ValueError(f"unsupported CandidateContext version: {version}")
    expected_width = candidate_context_feature_dim(2)
    if len(values) != expected_width:
        raise ValueError(
            f"CandidateContextV2 row must have width {expected_width}, got {len(values)}"
        )
    captured_piece_minor_or_major = max(values[17:21]) if values[15] > 0.0 else 0.0
    return [
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        values[13],
        values[14],
        values[15],
        values[16],
        float(captured_piece_minor_or_major),
    ]


def global_context_feature_order(
    version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
) -> tuple[str, ...]:
    """Return the feature order for one global-context version."""
    return _global_context_feature_order(version)


def global_context_feature_dim(version: int = DEFAULT_GLOBAL_CONTEXT_VERSION) -> int:
    """Return the feature width for one global-context version."""
    return len(_global_context_feature_order(version))


def transition_context_spec(version: int = 1) -> dict[str, object]:
    """Return the selected-action transition contract."""
    if version != 1:
        raise ValueError(f"unsupported TransitionContext version: {version}")
    candidate = candidate_context_spec(2)
    feature_order = transition_context_feature_order(version)
    return {
        "contract_name": "TransitionContext",
        "version": version,
        "candidate_context_version": candidate.version,
        "feature_dim": len(feature_order),
        "feature_order": list(feature_order),
        "post_move_feature_order": list(_TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER),
    }


def transition_context_feature_order(version: int = 1) -> tuple[str, ...]:
    """Return the feature order for one transition-context version."""
    if version != 1:
        raise ValueError(f"unsupported TransitionContext version: {version}")
    return _candidate_context_feature_order(2) + _TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER


def transition_context_feature_dim(version: int = 1) -> int:
    """Return the feature width for one transition-context version."""
    return len(transition_context_feature_order(version))


def symbolic_candidate_context_spec(
    *,
    candidate_context_version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
    global_context_version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
    max_legal_candidates: int = SYMBOLIC_MAX_LEGAL_CANDIDATES,
) -> dict[str, object]:
    """Return the symbolic proposer side-input contract."""
    candidate = candidate_context_spec(candidate_context_version)
    global_context = global_context_spec(global_context_version)
    return {
        "max_legal_candidates": max_legal_candidates,
        "candidate_context_version": candidate.version,
        "candidate_feature_dim": candidate.feature_dim,
        "candidate_feature_order": list(candidate.feature_order),
        "global_context_version": global_context.version,
        "global_feature_dim": global_context.feature_dim,
        "global_feature_order": list(global_context.feature_order),
    }


def _candidate_context_feature_order(version: int) -> tuple[str, ...]:
    if version == 1:
        return _CANDIDATE_CONTEXT_V1_FEATURE_ORDER
    if version == 2:
        return _CANDIDATE_CONTEXT_V2_FEATURE_ORDER
    raise ValueError(f"unsupported CandidateContext version: {version}")


def _global_context_feature_order(version: int) -> tuple[str, ...]:
    if version == 1:
        return _GLOBAL_CONTEXT_V1_FEATURE_ORDER
    raise ValueError(f"unsupported GlobalContext version: {version}")
