//! Exported-model loading and schema validation for runtime-facing inference
//! bundles.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const PROPOSER_SCHEMA_VERSION: u32 = 2;
pub const PROPOSER_METADATA_FILE: &str = "metadata.json";

/// Returns the current purpose of this crate.
pub fn crate_purpose() -> &'static str {
    "Rust-side loading and validation for exported proposer bundles"
}

/// Fully loaded proposer bundle with validated metadata and artifact paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerBundle {
    pub bundle_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub exported_program_path: PathBuf,
    pub metadata: ProposerMetadata,
}

/// Top-level proposer export metadata written by the Phase-5 Python exporter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerMetadata {
    pub schema_version: u32,
    pub model_name: String,
    pub artifacts: ProposerArtifacts,
    pub input: ProposerInputSpec,
    pub action_space: ProposerActionSpaceSpec,
    pub outputs: ProposerOutputSpec,
    pub training: ProposerTrainingSpec,
    pub validation_metrics: ProposerValidationMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposerArtifacts {
    pub checkpoint_file: String,
    pub exported_program_file: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerInputSpec {
    pub feature_dim: u32,
    pub layout: ProposerInputLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposerInputLayout {
    pub piece_token_capacity: u32,
    pub piece_token_width: u32,
    pub piece_padding_value: i32,
    pub square_token_count: u32,
    pub square_token_width: u32,
    pub rule_token_width: u32,
    pub flatten_order: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposerActionSpaceSpec {
    pub from_head_size: u32,
    pub to_head_size: u32,
    pub promotion_head_size: u32,
    pub flat_size: u32,
    pub flatten_formula: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerOutputSpec {
    pub legality_logits_shape: TensorShapeSpec,
    pub policy_logits_shape: TensorShapeSpec,
    pub legality_threshold: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShapeSpec {
    pub batch: String,
    pub actions: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerTrainingSpec {
    pub seed: u64,
    pub train_split: String,
    pub validation_split: String,
    pub hidden_dim: u32,
    pub hidden_layers: u32,
    pub dropout: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub legality_loss_weight: f32,
    pub policy_loss_weight: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerValidationMetrics {
    pub total_examples: u32,
    pub labeled_policy_examples: u32,
    pub total_loss: f32,
    pub legality_loss: f32,
    pub policy_loss: f32,
    pub legal_set_precision: f32,
    pub legal_set_recall: f32,
    pub legal_set_f1: f32,
    pub policy_top1_accuracy: f32,
}

/// Errors returned when loading an exported proposer bundle.
#[derive(Debug)]
pub enum ProposerLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
    MissingArtifact(PathBuf),
    InvalidFeatureDim {
        expected: u32,
        found: u32,
    },
    InvalidActionSpaceFlatSize {
        expected: u32,
        found: u32,
    },
    InvalidOutputShape {
        name: &'static str,
        actions: u32,
        expected: u32,
    },
    InvalidLegalityThreshold(f32),
}

impl fmt::Display for ProposerLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "could not read proposer bundle: {error}"),
            Self::Json(error) => write!(f, "invalid proposer metadata JSON: {error}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported proposer schema version: {version}")
            }
            Self::MissingArtifact(path) => write!(f, "required proposer artifact is missing: {path:?}"),
            Self::InvalidFeatureDim { expected, found } => {
                write!(f, "invalid proposer feature dim: expected {expected}, found {found}")
            }
            Self::InvalidActionSpaceFlatSize { expected, found } => {
                write!(
                    f,
                    "invalid proposer action-space size: expected {expected}, found {found}"
                )
            }
            Self::InvalidOutputShape {
                name,
                actions,
                expected,
            } => write!(
                f,
                "invalid proposer output shape for {name}: expected {expected} actions, found {actions}"
            ),
            Self::InvalidLegalityThreshold(threshold) => {
                write!(f, "invalid legality threshold: {threshold}")
            }
        }
    }
}

impl Error for ProposerLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion(_)
            | Self::MissingArtifact(_)
            | Self::InvalidFeatureDim { .. }
            | Self::InvalidActionSpaceFlatSize { .. }
            | Self::InvalidOutputShape { .. }
            | Self::InvalidLegalityThreshold(_) => None,
        }
    }
}

/// Load and validate a proposer export bundle from disk.
pub fn load_proposer_bundle(
    bundle_dir: impl AsRef<Path>,
) -> Result<ProposerBundle, ProposerLoadError> {
    let bundle_dir = bundle_dir.as_ref().to_path_buf();
    let metadata_path = bundle_dir.join(PROPOSER_METADATA_FILE);
    let metadata = load_proposer_metadata(&metadata_path)?;
    validate_proposer_metadata(&metadata)?;

    let checkpoint_path = bundle_dir.join(&metadata.artifacts.checkpoint_file);
    let exported_program_path = bundle_dir.join(&metadata.artifacts.exported_program_file);

    if !checkpoint_path.is_file() {
        return Err(ProposerLoadError::MissingArtifact(checkpoint_path));
    }
    if !exported_program_path.is_file() {
        return Err(ProposerLoadError::MissingArtifact(exported_program_path));
    }

    Ok(ProposerBundle {
        bundle_dir,
        checkpoint_path,
        exported_program_path,
        metadata,
    })
}

/// Load proposer metadata from a metadata JSON file.
pub fn load_proposer_metadata(
    path: impl AsRef<Path>,
) -> Result<ProposerMetadata, ProposerLoadError> {
    let path = path.as_ref();
    let payload = fs::read_to_string(path).map_err(ProposerLoadError::Io)?;
    serde_json::from_str(&payload).map_err(ProposerLoadError::Json)
}

/// Validate proposer metadata without touching the referenced artifact files.
pub fn validate_proposer_metadata(metadata: &ProposerMetadata) -> Result<(), ProposerLoadError> {
    if metadata.schema_version != PROPOSER_SCHEMA_VERSION {
        return Err(ProposerLoadError::UnsupportedSchemaVersion(
            metadata.schema_version,
        ));
    }

    let expected_feature_dim = metadata.input.layout.piece_token_capacity
        * metadata.input.layout.piece_token_width
        + metadata.input.layout.square_token_count * metadata.input.layout.square_token_width
        + metadata.input.layout.rule_token_width;
    if metadata.input.feature_dim != expected_feature_dim {
        return Err(ProposerLoadError::InvalidFeatureDim {
            expected: expected_feature_dim,
            found: metadata.input.feature_dim,
        });
    }

    let expected_flat_size = metadata.action_space.from_head_size
        * metadata.action_space.to_head_size
        * metadata.action_space.promotion_head_size;
    if metadata.action_space.flat_size != expected_flat_size {
        return Err(ProposerLoadError::InvalidActionSpaceFlatSize {
            expected: expected_flat_size,
            found: metadata.action_space.flat_size,
        });
    }

    validate_output_shape(
        "legality_logits_shape",
        &metadata.outputs.legality_logits_shape,
        metadata.action_space.flat_size,
    )?;
    validate_output_shape(
        "policy_logits_shape",
        &metadata.outputs.policy_logits_shape,
        metadata.action_space.flat_size,
    )?;

    if !(0.0..=1.0).contains(&metadata.outputs.legality_threshold) {
        return Err(ProposerLoadError::InvalidLegalityThreshold(
            metadata.outputs.legality_threshold,
        ));
    }

    Ok(())
}

fn validate_output_shape(
    name: &'static str,
    shape: &TensorShapeSpec,
    expected_actions: u32,
) -> Result<(), ProposerLoadError> {
    if shape.actions != expected_actions {
        return Err(ProposerLoadError::InvalidOutputShape {
            name,
            actions: shape.actions,
            expected: expected_actions,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        crate_purpose, load_proposer_bundle, validate_proposer_metadata, ProposerLoadError,
    };

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }

    #[test]
    fn load_proposer_bundle_accepts_valid_export() {
        let bundle_dir = tempdir().expect("temp dir");
        write_valid_bundle(bundle_dir.path());

        let bundle = load_proposer_bundle(bundle_dir.path()).expect("bundle loads");

        assert_eq!(bundle.metadata.schema_version, 2);
        assert_eq!(bundle.metadata.input.feature_dim, 230);
        assert_eq!(bundle.metadata.action_space.flat_size, 20_480);
        assert!(bundle.checkpoint_path.ends_with("checkpoint.pt"));
        assert!(bundle.exported_program_path.ends_with("proposer.pt2"));
    }

    #[test]
    fn validate_proposer_metadata_rejects_inconsistent_action_space() {
        let mut metadata = valid_metadata();
        metadata["action_space"]["flat_size"] = json!(7);

        let parsed = serde_json::from_value(metadata).expect("metadata parses");
        let error = validate_proposer_metadata(&parsed).expect_err("metadata should be rejected");

        assert!(matches!(
            error,
            ProposerLoadError::InvalidActionSpaceFlatSize {
                expected: 20_480,
                found: 7,
            }
        ));
    }

    #[test]
    fn load_proposer_bundle_requires_artifact_files() {
        let bundle_dir = tempdir().expect("temp dir");
        fs::write(
            bundle_dir.path().join("metadata.json"),
            serde_json::to_vec_pretty(&valid_metadata()).expect("serialize metadata"),
        )
        .expect("write metadata");

        let error = load_proposer_bundle(bundle_dir.path()).expect_err("missing files should fail");
        assert!(matches!(error, ProposerLoadError::MissingArtifact(_)));
    }

    fn write_valid_bundle(bundle_dir: &Path) {
        fs::write(
            bundle_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&valid_metadata()).expect("serialize metadata"),
        )
        .expect("write metadata");
        fs::write(bundle_dir.join("checkpoint.pt"), b"checkpoint").expect("write checkpoint");
        fs::write(bundle_dir.join("proposer.pt2"), b"exported program")
            .expect("write exported program");
    }

    fn valid_metadata() -> serde_json::Value {
        json!({
            "schema_version": 2,
            "model_name": "legality_policy_proposer_v1",
            "artifacts": {
                "checkpoint_file": "checkpoint.pt",
                "exported_program_file": "proposer.pt2"
            },
            "input": {
                "feature_dim": 230,
                "layout": {
                    "piece_token_capacity": 32,
                    "piece_token_width": 3,
                    "piece_padding_value": -1,
                    "square_token_count": 64,
                    "square_token_width": 2,
                    "rule_token_width": 6,
                    "flatten_order": [
                        "piece_tokens padded to 32 rows with [-1, -1, -1]",
                        "square_tokens[64][square_index, occupant_code]",
                        "rule_token[side_to_move, castling_bits, en_passant_square, halfmove_clock, fullmove_number, repetition_count]"
                    ]
                }
            },
            "action_space": {
                "from_head_size": 64,
                "to_head_size": 64,
                "promotion_head_size": 5,
                "flat_size": 20480,
                "flatten_formula": "((from_index * 64) + to_index) * 5 + promotion_index"
            },
            "outputs": {
                "legality_logits_shape": {
                    "batch": "dynamic",
                    "actions": 20480
                },
                "policy_logits_shape": {
                    "batch": "dynamic",
                    "actions": 20480
                },
                "legality_threshold": 0.5
            },
            "training": {
                "seed": 5,
                "train_split": "train",
                "validation_split": "validation",
                "hidden_dim": 128,
                "hidden_layers": 2,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "legality_loss_weight": 1.0,
                "policy_loss_weight": 1.0
            },
            "validation_metrics": {
                "total_examples": 1,
                "labeled_policy_examples": 1,
                "total_loss": 1.0,
                "legality_loss": 0.5,
                "policy_loss": 0.5,
                "legal_set_precision": 0.4,
                "legal_set_recall": 0.6,
                "legal_set_f1": 0.48,
                "policy_top1_accuracy": 1.0
            }
        })
    }
}
