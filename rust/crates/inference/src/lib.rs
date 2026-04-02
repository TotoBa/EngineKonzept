//! Exported-model loading and schema validation for runtime-facing inference
//! bundles.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use action_space::encode_move;
use core_types::{Color, Move, MoveKind, PieceKind, Square};
use encoder::{encode_position, EncodedPosition};
use position::Position;
use rules::{apply_move, is_in_check, is_square_attacked, legal_moves};
use serde::{Deserialize, Serialize};

pub const PROPOSER_SCHEMA_VERSION: u32 = 4;
pub const DYNAMICS_SCHEMA_VERSION: u32 = 1;
pub const PROPOSER_METADATA_FILE: &str = "metadata.json";
pub const SYMBOLIC_MAX_LEGAL_CANDIDATES: usize = 256;
pub const SYMBOLIC_CANDIDATE_FEATURE_DIM: usize = 18;
pub const SYMBOLIC_GLOBAL_FEATURE_DIM: usize = 9;
const SYMBOLIC_RUNTIME_MAGIC: &[u8; 8] = b"EKSPRT1\n";
const SYMBOLIC_RUNTIME_VERSION: u32 = 1;

/// Returns the current purpose of this crate.
pub fn crate_purpose() -> &'static str {
    "Rust-side loading and validation for exported proposer and dynamics bundles"
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
    #[serde(default)]
    pub runtime_weights_file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerInputSpec {
    pub feature_dim: u32,
    pub layout: ProposerInputLayout,
    #[serde(default)]
    pub symbolic: Option<ProposerSymbolicInputSpec>,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposerSymbolicInputSpec {
    pub max_legal_candidates: u32,
    pub candidate_feature_dim: u32,
    pub global_feature_dim: u32,
    pub candidate_feature_order: Vec<String>,
    pub global_feature_order: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposerOutputSpec {
    pub legality_logits_shape: TensorShapeSpec,
    pub policy_logits_shape: TensorShapeSpec,
    pub legality_threshold: f32,
    #[serde(default)]
    pub legality_source: Option<String>,
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

/// Fully loaded dynamics bundle with validated metadata and artifact paths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsBundle {
    pub bundle_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub exported_program_path: PathBuf,
    pub metadata: DynamicsMetadata,
}

/// Top-level dynamics export metadata written by the Phase-6 Python exporter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsMetadata {
    pub schema_version: u32,
    pub model_name: String,
    pub artifacts: ProposerArtifacts,
    pub input: DynamicsInputSpec,
    pub latent: DynamicsLatentSpec,
    pub outputs: DynamicsOutputSpec,
    pub training: DynamicsTrainingSpec,
    pub validation_metrics: DynamicsValidationMetrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsInputSpec {
    pub state: ProposerInputSpec,
    pub action: DynamicsActionInputSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicsActionInputSpec {
    pub from_head_size: u32,
    pub to_head_size: u32,
    pub promotion_head_size: u32,
    pub flat_size: u32,
    pub flatten_formula: String,
    pub dtype: String,
    pub shape: DynamicBatchShapeSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicBatchShapeSpec {
    pub batch: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicsLatentSpec {
    pub latent_dim: u32,
    pub action_embedding_dim: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsOutputSpec {
    pub next_state_shape: FeatureTensorShapeSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureTensorShapeSpec {
    pub batch: String,
    pub features: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsTrainingSpec {
    pub seed: u64,
    pub train_split: String,
    pub validation_split: String,
    pub latent_dim: u32,
    pub hidden_dim: u32,
    pub hidden_layers: u32,
    pub action_embedding_dim: u32,
    pub dropout: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub reconstruction_loss_weight: f32,
    pub drift_horizon: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicsValidationMetrics {
    pub total_examples: u32,
    pub total_loss: f32,
    pub reconstruction_loss: f32,
    pub feature_l1_error: f32,
    pub exact_next_feature_accuracy: f32,
    pub capture_examples: u32,
    pub capture_exact_next_feature_accuracy: f32,
    pub promotion_examples: u32,
    pub promotion_exact_next_feature_accuracy: f32,
    pub castle_examples: u32,
    pub castle_exact_next_feature_accuracy: f32,
    pub en_passant_examples: u32,
    pub en_passant_exact_next_feature_accuracy: f32,
    pub gives_check_examples: u32,
    pub gives_check_exact_next_feature_accuracy: f32,
    pub drift_examples: u32,
    pub drift_feature_l1_error: f32,
    pub drift_exact_next_feature_accuracy: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicProposerCandidate {
    pub chess_move: Move,
    pub move_uci: String,
    pub action_index: u32,
    pub features: [f32; SYMBOLIC_CANDIDATE_FEATURE_DIM],
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicProposerInputs {
    pub state_features: Vec<f32>,
    pub global_features: [f32; SYMBOLIC_GLOBAL_FEATURE_DIM],
    pub candidate_action_indices: Vec<i64>,
    pub candidate_features: Vec<[f32; SYMBOLIC_CANDIDATE_FEATURE_DIM]>,
    pub candidate_mask: Vec<bool>,
    pub candidates: Vec<SymbolicProposerCandidate>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicScoredCandidate {
    pub candidate: SymbolicProposerCandidate,
    pub policy_score: f32,
}

#[derive(Clone, Debug)]
pub struct SymbolicProposerRuntime {
    pub bundle: ProposerBundle,
    state_layers: Vec<LinearLayer>,
    global_projection: LinearLayer,
    action_embedding: EmbeddingLayer,
    candidate_linear_1: LinearLayer,
    candidate_linear_2: LinearLayer,
    candidate_linear_3: LinearLayer,
}

#[derive(Clone, Debug)]
struct LinearLayer {
    input_dim: usize,
    output_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone, Debug)]
struct EmbeddingLayer {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
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
    InvalidSymbolicMaxLegalCandidates(u32),
    InvalidSymbolicCandidateFeatureDim(u32),
    InvalidSymbolicGlobalFeatureDim(u32),
    InvalidLegalitySource(String),
    MissingRuntimeWeights(PathBuf),
    InvalidRuntimeWeights(String),
}

/// Errors returned when loading an exported dynamics bundle.
#[derive(Debug)]
pub enum DynamicsLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u32),
    MissingArtifact(PathBuf),
    InvalidFeatureDim { expected: u32, found: u32 },
    InvalidActionSpaceFlatSize { expected: u32, found: u32 },
    InvalidActionDtype(String),
    InvalidNextStateShape { expected: u32, found: u32 },
    InvalidDriftHorizon(u32),
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
            Self::InvalidSymbolicMaxLegalCandidates(value) => {
                write!(f, "invalid symbolic max_legal_candidates: {value}")
            }
            Self::InvalidSymbolicCandidateFeatureDim(value) => {
                write!(f, "invalid symbolic candidate_feature_dim: {value}")
            }
            Self::InvalidSymbolicGlobalFeatureDim(value) => {
                write!(f, "invalid symbolic global_feature_dim: {value}")
            }
            Self::InvalidLegalitySource(source) => {
                write!(f, "invalid proposer legality_source: {source}")
            }
            Self::MissingRuntimeWeights(path) => {
                write!(f, "required proposer runtime weights are missing: {path:?}")
            }
            Self::InvalidRuntimeWeights(message) => {
                write!(f, "invalid proposer runtime weights: {message}")
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
            | Self::InvalidLegalityThreshold(_)
            | Self::InvalidSymbolicMaxLegalCandidates(_)
            | Self::InvalidSymbolicCandidateFeatureDim(_)
            | Self::InvalidSymbolicGlobalFeatureDim(_)
            | Self::InvalidLegalitySource(_)
            | Self::MissingRuntimeWeights(_)
            | Self::InvalidRuntimeWeights(_) => None,
        }
    }
}

impl fmt::Display for DynamicsLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "could not read dynamics bundle: {error}"),
            Self::Json(error) => write!(f, "invalid dynamics metadata JSON: {error}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported dynamics schema version: {version}")
            }
            Self::MissingArtifact(path) => {
                write!(f, "required dynamics artifact is missing: {path:?}")
            }
            Self::InvalidFeatureDim { expected, found } => {
                write!(
                    f,
                    "invalid dynamics feature dim: expected {expected}, found {found}"
                )
            }
            Self::InvalidActionSpaceFlatSize { expected, found } => write!(
                f,
                "invalid dynamics action-space size: expected {expected}, found {found}"
            ),
            Self::InvalidActionDtype(dtype) => {
                write!(
                    f,
                    "invalid dynamics action dtype: expected int64, found {dtype}"
                )
            }
            Self::InvalidNextStateShape { expected, found } => write!(
                f,
                "invalid dynamics next-state shape: expected {expected} features, found {found}"
            ),
            Self::InvalidDriftHorizon(horizon) => {
                write!(f, "invalid dynamics drift horizon: {horizon}")
            }
        }
    }
}

impl Error for DynamicsLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion(_)
            | Self::MissingArtifact(_)
            | Self::InvalidFeatureDim { .. }
            | Self::InvalidActionSpaceFlatSize { .. }
            | Self::InvalidActionDtype(_)
            | Self::InvalidNextStateShape { .. }
            | Self::InvalidDriftHorizon(_) => None,
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
    let runtime_weights_path = metadata
        .artifacts
        .runtime_weights_file
        .as_ref()
        .map(|file_name| bundle_dir.join(file_name));

    if !checkpoint_path.is_file() {
        return Err(ProposerLoadError::MissingArtifact(checkpoint_path));
    }
    if !exported_program_path.is_file() {
        return Err(ProposerLoadError::MissingArtifact(exported_program_path));
    }
    if let Some(path) = runtime_weights_path {
        if !path.is_file() {
            return Err(ProposerLoadError::MissingRuntimeWeights(path));
        }
    }

    Ok(ProposerBundle {
        bundle_dir,
        checkpoint_path,
        exported_program_path,
        metadata,
    })
}

/// Load and validate a dynamics export bundle from disk.
pub fn load_dynamics_bundle(
    bundle_dir: impl AsRef<Path>,
) -> Result<DynamicsBundle, DynamicsLoadError> {
    let bundle_dir = bundle_dir.as_ref().to_path_buf();
    let metadata_path = bundle_dir.join(PROPOSER_METADATA_FILE);
    let metadata = load_dynamics_metadata(&metadata_path)?;
    validate_dynamics_metadata(&metadata)?;

    let checkpoint_path = bundle_dir.join(&metadata.artifacts.checkpoint_file);
    let exported_program_path = bundle_dir.join(&metadata.artifacts.exported_program_file);

    if !checkpoint_path.is_file() {
        return Err(DynamicsLoadError::MissingArtifact(checkpoint_path));
    }
    if !exported_program_path.is_file() {
        return Err(DynamicsLoadError::MissingArtifact(exported_program_path));
    }

    Ok(DynamicsBundle {
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

/// Load dynamics metadata from a metadata JSON file.
pub fn load_dynamics_metadata(
    path: impl AsRef<Path>,
) -> Result<DynamicsMetadata, DynamicsLoadError> {
    let path = path.as_ref();
    let payload = fs::read_to_string(path).map_err(DynamicsLoadError::Io)?;
    serde_json::from_str(&payload).map_err(DynamicsLoadError::Json)
}

/// Validate proposer metadata without touching the referenced artifact files.
pub fn validate_proposer_metadata(metadata: &ProposerMetadata) -> Result<(), ProposerLoadError> {
    if !(2..=PROPOSER_SCHEMA_VERSION).contains(&metadata.schema_version) {
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

    if let Some(symbolic) = &metadata.input.symbolic {
        if symbolic.max_legal_candidates == 0
            || symbolic.max_legal_candidates > SYMBOLIC_MAX_LEGAL_CANDIDATES as u32
        {
            return Err(ProposerLoadError::InvalidSymbolicMaxLegalCandidates(
                symbolic.max_legal_candidates,
            ));
        }
        if symbolic.candidate_feature_dim != SYMBOLIC_CANDIDATE_FEATURE_DIM as u32 {
            return Err(ProposerLoadError::InvalidSymbolicCandidateFeatureDim(
                symbolic.candidate_feature_dim,
            ));
        }
        if symbolic.global_feature_dim != SYMBOLIC_GLOBAL_FEATURE_DIM as u32 {
            return Err(ProposerLoadError::InvalidSymbolicGlobalFeatureDim(
                symbolic.global_feature_dim,
            ));
        }
    }

    match metadata
        .outputs
        .legality_source
        .as_deref()
        .unwrap_or("learned_head")
    {
        "learned_head" | "symbolic_generator" => {}
        other => return Err(ProposerLoadError::InvalidLegalitySource(other.to_string())),
    }

    Ok(())
}

pub fn load_symbolic_proposer_runtime(
    bundle_dir: impl AsRef<Path>,
) -> Result<SymbolicProposerRuntime, ProposerLoadError> {
    let bundle = load_proposer_bundle(bundle_dir)?;
    if bundle.metadata.outputs.legality_source.as_deref() != Some("symbolic_generator") {
        return Err(ProposerLoadError::InvalidLegalitySource(
            bundle
                .metadata
                .outputs
                .legality_source
                .clone()
                .unwrap_or_else(|| "learned_head".to_string()),
        ));
    }
    let runtime_weights_file = bundle
        .metadata
        .artifacts
        .runtime_weights_file
        .as_ref()
        .ok_or_else(|| {
            ProposerLoadError::InvalidRuntimeWeights(
                "symbolic bundle metadata is missing runtime_weights_file".to_string(),
            )
        })?;
    let runtime_path = bundle.bundle_dir.join(runtime_weights_file);
    let weights = fs::read(runtime_path).map_err(ProposerLoadError::Io)?;
    parse_symbolic_runtime_weights(&bundle, &weights)
}

/// Validate dynamics metadata without touching the referenced artifact files.
pub fn validate_dynamics_metadata(metadata: &DynamicsMetadata) -> Result<(), DynamicsLoadError> {
    if metadata.schema_version != DYNAMICS_SCHEMA_VERSION {
        return Err(DynamicsLoadError::UnsupportedSchemaVersion(
            metadata.schema_version,
        ));
    }

    let expected_feature_dim = metadata.input.state.layout.piece_token_capacity
        * metadata.input.state.layout.piece_token_width
        + metadata.input.state.layout.square_token_count
            * metadata.input.state.layout.square_token_width
        + metadata.input.state.layout.rule_token_width;
    if metadata.input.state.feature_dim != expected_feature_dim {
        return Err(DynamicsLoadError::InvalidFeatureDim {
            expected: expected_feature_dim,
            found: metadata.input.state.feature_dim,
        });
    }

    let expected_flat_size = metadata.input.action.from_head_size
        * metadata.input.action.to_head_size
        * metadata.input.action.promotion_head_size;
    if metadata.input.action.flat_size != expected_flat_size {
        return Err(DynamicsLoadError::InvalidActionSpaceFlatSize {
            expected: expected_flat_size,
            found: metadata.input.action.flat_size,
        });
    }

    if metadata.input.action.dtype != "int64" {
        return Err(DynamicsLoadError::InvalidActionDtype(
            metadata.input.action.dtype.clone(),
        ));
    }

    if metadata.outputs.next_state_shape.features != metadata.input.state.feature_dim {
        return Err(DynamicsLoadError::InvalidNextStateShape {
            expected: metadata.input.state.feature_dim,
            found: metadata.outputs.next_state_shape.features,
        });
    }

    if metadata.training.drift_horizon < 2 {
        return Err(DynamicsLoadError::InvalidDriftHorizon(
            metadata.training.drift_horizon,
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

impl SymbolicProposerRuntime {
    pub fn score_position(
        &self,
        position: &Position,
    ) -> Result<Vec<SymbolicScoredCandidate>, action_space::ActionEncodeError> {
        let inputs = build_symbolic_proposer_inputs(position)?;
        Ok(self.score_inputs(inputs))
    }

    pub fn score_inputs(&self, inputs: SymbolicProposerInputs) -> Vec<SymbolicScoredCandidate> {
        let hidden = run_mlp_layers(&self.state_layers, &inputs.state_features);
        let mut global_input = Vec::with_capacity(hidden.len() + inputs.global_features.len());
        global_input.extend_from_slice(&hidden);
        global_input.extend_from_slice(&inputs.global_features);
        let global_hidden = relu(self.global_projection.forward(&global_input));

        let mut scored = Vec::with_capacity(inputs.candidates.len());
        for candidate in inputs.candidates {
            let action_embedding = self.action_embedding.row(candidate.action_index as usize);
            let mut candidate_input = Vec::with_capacity(
                global_hidden.len() + action_embedding.len() + candidate.features.len(),
            );
            candidate_input.extend_from_slice(&global_hidden);
            candidate_input.extend_from_slice(action_embedding);
            candidate_input.extend_from_slice(&candidate.features);
            let hidden_1 = relu(self.candidate_linear_1.forward(&candidate_input));
            let hidden_2 = relu(self.candidate_linear_2.forward(&hidden_1));
            let policy_score = self.candidate_linear_3.forward(&hidden_2)[0];
            scored.push(SymbolicScoredCandidate {
                candidate,
                policy_score,
            });
        }

        scored
    }
}

impl LinearLayer {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_dim);
        let mut output = self.bias.clone();
        debug_assert_eq!(output.len(), self.output_dim);
        for (row_index, row) in self.weight.chunks_exact(self.input_dim).enumerate() {
            output[row_index] += dot(row, input);
        }
        output
    }
}

impl EmbeddingLayer {
    fn row(&self, index: usize) -> &[f32] {
        debug_assert!(index < self.rows);
        let start = index * self.cols;
        &self.values[start..start + self.cols]
    }
}

fn parse_symbolic_runtime_weights(
    bundle: &ProposerBundle,
    bytes: &[u8],
) -> Result<SymbolicProposerRuntime, ProposerLoadError> {
    let hidden_dim = bundle.metadata.training.hidden_dim as usize;
    let hidden_layers = bundle.metadata.training.hidden_layers as usize;
    let half_hidden_dim = hidden_dim / 2;
    let mut cursor = 0;

    let magic = take_bytes(bytes, &mut cursor, SYMBOLIC_RUNTIME_MAGIC.len())?;
    if magic != SYMBOLIC_RUNTIME_MAGIC {
        return Err(ProposerLoadError::InvalidRuntimeWeights(
            "unexpected symbolic runtime magic".to_string(),
        ));
    }

    let runtime_version = take_u32(bytes, &mut cursor)?;
    if runtime_version != SYMBOLIC_RUNTIME_VERSION {
        return Err(ProposerLoadError::InvalidRuntimeWeights(format!(
            "unsupported symbolic runtime version: {runtime_version}"
        )));
    }

    let file_hidden_dim = take_u32(bytes, &mut cursor)? as usize;
    let file_hidden_layers = take_u32(bytes, &mut cursor)? as usize;
    let action_embedding_dim = take_u32(bytes, &mut cursor)? as usize;
    if file_hidden_dim != hidden_dim {
        return Err(ProposerLoadError::InvalidRuntimeWeights(format!(
            "runtime hidden_dim mismatch: expected {hidden_dim}, found {file_hidden_dim}"
        )));
    }
    if file_hidden_layers != hidden_layers {
        return Err(ProposerLoadError::InvalidRuntimeWeights(format!(
            "runtime hidden_layers mismatch: expected {hidden_layers}, found {file_hidden_layers}"
        )));
    }

    let mut state_layers = Vec::with_capacity(hidden_layers);
    let mut input_dim = bundle.metadata.input.feature_dim as usize;
    for _ in 0..hidden_layers {
        state_layers.push(LinearLayer {
            input_dim,
            output_dim: hidden_dim,
            weight: take_f32s(bytes, &mut cursor, hidden_dim * input_dim)?,
            bias: take_f32s(bytes, &mut cursor, hidden_dim)?,
        });
        input_dim = hidden_dim;
    }

    let global_projection = LinearLayer {
        input_dim: hidden_dim + SYMBOLIC_GLOBAL_FEATURE_DIM,
        output_dim: hidden_dim,
        weight: take_f32s(
            bytes,
            &mut cursor,
            hidden_dim * (hidden_dim + SYMBOLIC_GLOBAL_FEATURE_DIM),
        )?,
        bias: take_f32s(bytes, &mut cursor, hidden_dim)?,
    };
    let action_embedding = EmbeddingLayer {
        rows: bundle.metadata.action_space.flat_size as usize,
        cols: action_embedding_dim,
        values: take_f32s(
            bytes,
            &mut cursor,
            bundle.metadata.action_space.flat_size as usize * action_embedding_dim,
        )?,
    };
    let candidate_linear_1 = LinearLayer {
        input_dim: hidden_dim + action_embedding_dim + SYMBOLIC_CANDIDATE_FEATURE_DIM,
        output_dim: hidden_dim,
        weight: take_f32s(
            bytes,
            &mut cursor,
            hidden_dim * (hidden_dim + action_embedding_dim + SYMBOLIC_CANDIDATE_FEATURE_DIM),
        )?,
        bias: take_f32s(bytes, &mut cursor, hidden_dim)?,
    };
    let candidate_linear_2 = LinearLayer {
        input_dim: hidden_dim,
        output_dim: half_hidden_dim,
        weight: take_f32s(bytes, &mut cursor, hidden_dim * half_hidden_dim)?,
        bias: take_f32s(bytes, &mut cursor, half_hidden_dim)?,
    };
    let candidate_linear_3 = LinearLayer {
        input_dim: half_hidden_dim,
        output_dim: 1,
        weight: take_f32s(bytes, &mut cursor, half_hidden_dim)?,
        bias: take_f32s(bytes, &mut cursor, 1)?,
    };

    if cursor != bytes.len() {
        return Err(ProposerLoadError::InvalidRuntimeWeights(
            "runtime weights have trailing bytes".to_string(),
        ));
    }

    Ok(SymbolicProposerRuntime {
        bundle: bundle.clone(),
        state_layers,
        global_projection,
        action_embedding,
        candidate_linear_1,
        candidate_linear_2,
        candidate_linear_3,
    })
}

fn run_mlp_layers(layers: &[LinearLayer], input: &[f32]) -> Vec<f32> {
    let mut hidden = input.to_vec();
    for layer in layers {
        hidden = relu(layer.forward(&hidden));
    }
    hidden
}

fn relu(mut values: Vec<f32>) -> Vec<f32> {
    for value in &mut values {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    values
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>()
}

fn take_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    count: usize,
) -> Result<&'a [u8], ProposerLoadError> {
    let end = cursor.saturating_add(count);
    if end > bytes.len() {
        return Err(ProposerLoadError::InvalidRuntimeWeights(
            "runtime weights truncated".to_string(),
        ));
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, ProposerLoadError> {
    let raw = take_bytes(bytes, cursor, std::mem::size_of::<u32>())?;
    let mut array = [0_u8; std::mem::size_of::<u32>()];
    array.copy_from_slice(raw);
    Ok(u32::from_le_bytes(array))
}

fn take_f32s(
    bytes: &[u8],
    cursor: &mut usize,
    count: usize,
) -> Result<Vec<f32>, ProposerLoadError> {
    let raw = take_bytes(bytes, cursor, count * std::mem::size_of::<f32>())?;
    let mut values = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        let mut array = [0_u8; std::mem::size_of::<f32>()];
        array.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(array));
    }
    Ok(values)
}

/// Build the official symbolic proposer input contract for one exact position.
pub fn build_symbolic_proposer_inputs(
    position: &Position,
) -> Result<SymbolicProposerInputs, action_space::ActionEncodeError> {
    let encoded = encode_position(position);
    let state_features = pack_position_features(&encoded);
    let legal = legal_moves(position);
    let own_attacks = attack_map(position, position.side_to_move());
    let opponent = match position.side_to_move() {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let opponent_attacks = attack_map(position, opponent);

    let mut candidates = Vec::with_capacity(legal.len());
    for chess_move in legal {
        let action_index = flatten_action_index(encode_move(chess_move)?);
        let move_uci = chess_move.to_uci();
        let features =
            build_candidate_features(position, chess_move, &own_attacks, &opponent_attacks);
        candidates.push(SymbolicProposerCandidate {
            chess_move,
            move_uci,
            action_index,
            features,
        });
    }
    candidates.sort_by_key(|candidate| candidate.action_index);

    let mut candidate_action_indices = vec![-1_i64; SYMBOLIC_MAX_LEGAL_CANDIDATES];
    let mut candidate_features =
        vec![[0.0; SYMBOLIC_CANDIDATE_FEATURE_DIM]; SYMBOLIC_MAX_LEGAL_CANDIDATES];
    let mut candidate_mask = vec![false; SYMBOLIC_MAX_LEGAL_CANDIDATES];
    for (index, candidate) in candidates.iter().enumerate() {
        if index >= SYMBOLIC_MAX_LEGAL_CANDIDATES {
            break;
        }
        candidate_action_indices[index] = i64::from(candidate.action_index);
        candidate_features[index] = candidate.features;
        candidate_mask[index] = true;
    }

    Ok(SymbolicProposerInputs {
        state_features,
        global_features: build_global_features(
            position,
            candidates.len(),
            &own_attacks,
            &opponent_attacks,
        ),
        candidate_action_indices,
        candidate_features,
        candidate_mask,
        candidates,
    })
}

fn pack_position_features(encoded: &EncodedPosition) -> Vec<f32> {
    let mut features = Vec::with_capacity(
        (encoded.piece_tokens.len() * 3) + (64 * 2) + 6 + ((32 - encoded.piece_tokens.len()) * 3),
    );
    for token in &encoded.piece_tokens {
        for value in token.as_array() {
            features.push(value as f32);
        }
    }
    for _ in encoded.piece_tokens.len()..32 {
        features.extend_from_slice(&[-1.0, -1.0, -1.0]);
    }
    for token in encoded.square_token_matrix() {
        for value in token {
            features.push(value as f32);
        }
    }
    for value in encoded.rule_token_vector() {
        features.push(value as f32);
    }
    features
}

fn build_global_features(
    position: &Position,
    legal_move_count: usize,
    own_attacks: &[bool; 64],
    opponent_attacks: &[bool; 64],
) -> [f32; SYMBOLIC_GLOBAL_FEATURE_DIM] {
    let legal = legal_moves(position);
    let piece_count = position.iter_pieces().count();
    [
        bool_to_f32(is_in_check(position, position.side_to_move())),
        bool_to_f32(legal.iter().any(|mv| {
            matches!(
                mv.kind,
                MoveKind::CastleKingside | MoveKind::CastleQueenside
            )
        })),
        bool_to_f32(
            legal
                .iter()
                .any(|mv| matches!(mv.kind, MoveKind::EnPassant)),
        ),
        bool_to_f32(legal.iter().any(|mv| mv.kind.promotion_piece().is_some())),
        bool_to_f32(piece_count <= 8),
        (legal_move_count as f32) / 256.0,
        (piece_count as f32) / 32.0,
        (own_attacks.iter().filter(|attacked| **attacked).count() as f32) / 64.0,
        (opponent_attacks
            .iter()
            .filter(|attacked| **attacked)
            .count() as f32)
            / 64.0,
    ]
}

fn build_candidate_features(
    position: &Position,
    chess_move: Move,
    own_attacks: &[bool; 64],
    opponent_attacks: &[bool; 64],
) -> [f32; SYMBOLIC_CANDIDATE_FEATURE_DIM] {
    let moving_piece = position.board()[usize::from(chess_move.from.index())]
        .expect("legal move source must contain a piece");
    let captured_piece = captured_piece_for_move(position, chess_move);
    let next_position = apply_move(position, chess_move).expect("legal move must apply");
    let piece_type = moving_piece.kind;
    [
        bool_to_f32(chess_move.kind.is_capture()),
        bool_to_f32(chess_move.kind.promotion_piece().is_some()),
        bool_to_f32(matches!(
            chess_move.kind,
            MoveKind::CastleKingside | MoveKind::CastleQueenside
        )),
        bool_to_f32(matches!(chess_move.kind, MoveKind::EnPassant)),
        bool_to_f32(is_in_check(&next_position, next_position.side_to_move())),
        bool_to_f32(opponent_attacks[usize::from(chess_move.from.index())]),
        bool_to_f32(opponent_attacks[usize::from(chess_move.to.index())]),
        bool_to_f32(own_attacks[usize::from(chess_move.from.index())]),
        bool_to_f32(own_attacks[usize::from(chess_move.to.index())]),
        bool_to_f32(piece_type == PieceKind::Pawn),
        bool_to_f32(piece_type == PieceKind::Knight),
        bool_to_f32(piece_type == PieceKind::Bishop),
        bool_to_f32(piece_type == PieceKind::Rook),
        bool_to_f32(piece_type == PieceKind::Queen),
        bool_to_f32(piece_type == PieceKind::King),
        bool_to_f32(captured_piece.is_some()),
        bool_to_f32(matches!(captured_piece, Some(piece) if piece.kind == PieceKind::Pawn)),
        bool_to_f32(matches!(
            captured_piece,
            Some(piece)
                if matches!(
                    piece.kind,
                    PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen
                )
        )),
    ]
}

fn attack_map(position: &Position, attacker: Color) -> [bool; 64] {
    let mut attacked = [false; 64];
    for index in 0..64_u8 {
        let square = Square::new(index).expect("validated square index");
        attacked[usize::from(index)] = is_square_attacked(position, square, attacker);
    }
    attacked
}

fn flatten_action_index(action: action_space::ActionEncoding) -> u32 {
    let [from_index, to_index, promotion_index] = action.as_indices();
    (((from_index * 64) + to_index) * 5 + promotion_index) as u32
}

fn bool_to_f32(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn captured_piece_for_move(position: &Position, chess_move: Move) -> Option<core_types::Piece> {
    match chess_move.kind {
        MoveKind::EnPassant => {
            let rank_delta = if position.side_to_move() == Color::White {
                -1
            } else {
                1
            };
            chess_move
                .to
                .offset(0, rank_delta)
                .and_then(|square| position.board()[usize::from(square.index())])
        }
        _ => position.board()[usize::from(chess_move.to.index())],
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use action_space::encode_move;
    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        build_symbolic_proposer_inputs, crate_purpose, flatten_action_index, load_dynamics_bundle,
        load_proposer_bundle, load_symbolic_proposer_runtime, validate_dynamics_metadata,
        validate_proposer_metadata, DynamicsLoadError, ProposerLoadError,
        SYMBOLIC_CANDIDATE_FEATURE_DIM, SYMBOLIC_GLOBAL_FEATURE_DIM, SYMBOLIC_MAX_LEGAL_CANDIDATES,
        SYMBOLIC_RUNTIME_MAGIC, SYMBOLIC_RUNTIME_VERSION,
    };
    use position::Position;
    use rules::legal_moves;

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }

    #[test]
    fn load_proposer_bundle_accepts_valid_export() {
        let bundle_dir = tempdir().expect("temp dir");
        write_valid_bundle(bundle_dir.path());

        let bundle = load_proposer_bundle(bundle_dir.path()).expect("bundle loads");

        assert_eq!(bundle.metadata.schema_version, 4);
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
    fn build_symbolic_inputs_matches_runtime_contract() {
        let position = Position::startpos();
        let inputs = build_symbolic_proposer_inputs(&position).expect("symbolic inputs build");

        assert_eq!(inputs.state_features.len(), 230);
        assert_eq!(inputs.global_features.len(), SYMBOLIC_GLOBAL_FEATURE_DIM);
        assert_eq!(
            inputs.candidate_action_indices.len(),
            SYMBOLIC_MAX_LEGAL_CANDIDATES
        );
        assert_eq!(
            inputs.candidate_features.len(),
            SYMBOLIC_MAX_LEGAL_CANDIDATES
        );
        assert_eq!(inputs.candidates.len(), 20);
        assert_eq!(
            inputs.candidate_features[0].len(),
            SYMBOLIC_CANDIDATE_FEATURE_DIM
        );
        assert!(inputs.candidate_mask.iter().filter(|value| **value).count() == 20);
    }

    #[test]
    fn symbolic_runtime_scores_preferred_legal_move() {
        let bundle_dir = tempdir().expect("temp dir");
        write_valid_symbolic_runtime_bundle(bundle_dir.path(), "e2e4");

        let runtime = load_symbolic_proposer_runtime(bundle_dir.path()).expect("runtime loads");
        let scored = runtime
            .score_position(&Position::startpos())
            .expect("position scores");
        let best = scored
            .iter()
            .max_by(|left, right| left.policy_score.total_cmp(&right.policy_score))
            .expect("non-empty scores");

        assert_eq!(best.candidate.move_uci, "e2e4");
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

    #[test]
    fn load_dynamics_bundle_accepts_valid_export() {
        let bundle_dir = tempdir().expect("temp dir");
        write_valid_dynamics_bundle(bundle_dir.path());

        let bundle = load_dynamics_bundle(bundle_dir.path()).expect("bundle loads");

        assert_eq!(bundle.metadata.schema_version, 1);
        assert_eq!(bundle.metadata.input.state.feature_dim, 230);
        assert_eq!(bundle.metadata.latent.latent_dim, 128);
        assert!(bundle.exported_program_path.ends_with("dynamics.pt2"));
    }

    #[test]
    fn validate_dynamics_metadata_rejects_invalid_drift_horizon() {
        let mut metadata = valid_dynamics_metadata();
        metadata["training"]["drift_horizon"] = json!(1);

        let parsed = serde_json::from_value(metadata).expect("metadata parses");
        let error = validate_dynamics_metadata(&parsed).expect_err("metadata should be rejected");

        assert!(matches!(error, DynamicsLoadError::InvalidDriftHorizon(1)));
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

    fn write_valid_symbolic_runtime_bundle(bundle_dir: &Path, preferred_move_uci: &str) {
        let mut metadata = valid_metadata();
        metadata["input"]["symbolic"] = json!({
            "max_legal_candidates": SYMBOLIC_MAX_LEGAL_CANDIDATES,
            "candidate_feature_dim": SYMBOLIC_CANDIDATE_FEATURE_DIM,
            "global_feature_dim": SYMBOLIC_GLOBAL_FEATURE_DIM,
            "candidate_feature_order": [],
            "global_feature_order": []
        });
        metadata["outputs"]["legality_source"] = json!("symbolic_generator");
        metadata["artifacts"]["runtime_weights_file"] = json!("symbolic_runtime.bin");
        metadata["training"]["hidden_dim"] = json!(32);
        metadata["training"]["hidden_layers"] = json!(1);
        fs::write(
            bundle_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");
        fs::write(bundle_dir.join("checkpoint.pt"), b"checkpoint").expect("write checkpoint");
        fs::write(bundle_dir.join("proposer.pt2"), b"exported program")
            .expect("write exported program");
        fs::write(
            bundle_dir.join("symbolic_runtime.bin"),
            build_test_symbolic_runtime_weights(preferred_move_uci),
        )
        .expect("write symbolic runtime");
    }

    fn write_valid_dynamics_bundle(bundle_dir: &Path) {
        fs::write(
            bundle_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&valid_dynamics_metadata()).expect("serialize metadata"),
        )
        .expect("write metadata");
        fs::write(bundle_dir.join("checkpoint.pt"), b"checkpoint").expect("write checkpoint");
        fs::write(bundle_dir.join("dynamics.pt2"), b"exported program")
            .expect("write exported program");
    }

    fn valid_metadata() -> serde_json::Value {
        json!({
            "schema_version": 4,
            "model_name": "legality_policy_proposer_v1",
            "artifacts": {
                "checkpoint_file": "checkpoint.pt",
                "exported_program_file": "proposer.pt2",
                "runtime_weights_file": null
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
                },
                "symbolic": null
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
                "legality_threshold": 0.5,
                "legality_source": "learned_head"
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

    fn valid_dynamics_metadata() -> serde_json::Value {
        json!({
            "schema_version": 1,
            "model_name": "latent_dynamics_v1",
            "artifacts": {
                "checkpoint_file": "checkpoint.pt",
                "exported_program_file": "dynamics.pt2"
            },
            "input": {
                "state": {
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
                "action": {
                    "from_head_size": 64,
                    "to_head_size": 64,
                    "promotion_head_size": 5,
                    "flat_size": 20480,
                    "flatten_formula": "((from_index * 64) + to_index) * 5 + promotion_index",
                    "dtype": "int64",
                    "shape": {
                        "batch": "dynamic"
                    }
                }
            },
            "latent": {
                "latent_dim": 128,
                "action_embedding_dim": 64
            },
            "outputs": {
                "next_state_shape": {
                    "batch": "dynamic",
                    "features": 230
                }
            },
            "training": {
                "seed": 5,
                "train_split": "train",
                "validation_split": "validation",
                "latent_dim": 128,
                "hidden_dim": 256,
                "hidden_layers": 2,
                "action_embedding_dim": 64,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "reconstruction_loss_weight": 1.0,
                "drift_horizon": 2
            },
            "validation_metrics": {
                "total_examples": 4,
                "total_loss": 1.0,
                "reconstruction_loss": 1.0,
                "feature_l1_error": 0.5,
                "exact_next_feature_accuracy": 0.25,
                "capture_examples": 1,
                "capture_exact_next_feature_accuracy": 0.0,
                "promotion_examples": 1,
                "promotion_exact_next_feature_accuracy": 0.0,
                "castle_examples": 1,
                "castle_exact_next_feature_accuracy": 1.0,
                "en_passant_examples": 1,
                "en_passant_exact_next_feature_accuracy": 1.0,
                "gives_check_examples": 1,
                "gives_check_exact_next_feature_accuracy": 0.0,
                "drift_examples": 2,
                "drift_feature_l1_error": 0.75,
                "drift_exact_next_feature_accuracy": 0.0
            }
        })
    }

    fn build_test_symbolic_runtime_weights(preferred_move_uci: &str) -> Vec<u8> {
        let hidden_dim = 32_u32;
        let hidden_layers = 1_u32;
        let action_embedding_dim = 32_u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SYMBOLIC_RUNTIME_MAGIC);
        bytes.extend_from_slice(&SYMBOLIC_RUNTIME_VERSION.to_le_bytes());
        bytes.extend_from_slice(&hidden_dim.to_le_bytes());
        bytes.extend_from_slice(&hidden_layers.to_le_bytes());
        bytes.extend_from_slice(&action_embedding_dim.to_le_bytes());

        push_zeros(&mut bytes, (hidden_dim as usize) * 230);
        push_zeros(&mut bytes, hidden_dim as usize);
        push_zeros(
            &mut bytes,
            (hidden_dim as usize) * (hidden_dim as usize + SYMBOLIC_GLOBAL_FEATURE_DIM),
        );
        push_zeros(&mut bytes, hidden_dim as usize);

        let mut embedding = vec![0.0_f32; 20_480 * action_embedding_dim as usize];
        let preferred_move = legal_moves(&Position::startpos())
            .into_iter()
            .find(|candidate| candidate.to_uci() == preferred_move_uci)
            .expect("preferred move is legal in startpos");
        let preferred_index =
            flatten_action_index(encode_move(preferred_move).expect("move encodes"));
        embedding[preferred_index as usize * action_embedding_dim as usize] = 2.0;
        push_f32s(&mut bytes, &embedding);

        let candidate_input_dim =
            hidden_dim as usize + action_embedding_dim as usize + SYMBOLIC_CANDIDATE_FEATURE_DIM;
        let mut linear_1 = vec![0.0_f32; hidden_dim as usize * candidate_input_dim];
        linear_1[hidden_dim as usize] = 1.0;
        push_f32s(&mut bytes, &linear_1);
        push_zeros(&mut bytes, hidden_dim as usize);

        let mut linear_2 = vec![0.0_f32; (hidden_dim as usize / 2) * hidden_dim as usize];
        linear_2[0] = 1.0;
        push_f32s(&mut bytes, &linear_2);
        push_zeros(&mut bytes, hidden_dim as usize / 2);

        let mut linear_3 = vec![0.0_f32; hidden_dim as usize / 2];
        linear_3[0] = 1.0;
        push_f32s(&mut bytes, &linear_3);
        push_f32s(&mut bytes, &[0.0]);
        bytes
    }

    fn push_zeros(bytes: &mut Vec<u8>, count: usize) {
        for _ in 0..count {
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
    }

    fn push_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}
