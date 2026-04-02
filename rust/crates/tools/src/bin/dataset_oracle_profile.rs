use std::io;
use std::time::Duration;

use serde::Serialize;
use tools::{profile_json_lines, OracleProfileTotals};

#[derive(Serialize)]
struct ProfilePhaseReport {
    seconds: f64,
    milliseconds_per_record: f64,
    share_of_measured: f64,
}

#[derive(Serialize)]
struct JsonSectionReport {
    bytes: u64,
    bytes_per_record: f64,
    share_of_json_bytes: f64,
}

#[derive(Serialize)]
struct ProfileReport {
    records: u64,
    total_measured_seconds: f64,
    phases: Vec<(&'static str, ProfilePhaseReport)>,
    json_sections: Vec<(&'static str, JsonSectionReport)>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let profile = profile_json_lines(stdin.lock(), "dataset-oracle-profile")?;
    serde_json::to_writer_pretty(io::stdout().lock(), &to_report(&profile))?;
    println!();
    Ok(())
}

fn to_report(profile: &OracleProfileTotals) -> ProfileReport {
    let total_measured = profile.total_measured();
    let records = profile.records.max(1);

    ProfileReport {
        records: profile.records,
        total_measured_seconds: seconds(total_measured),
        phases: vec![
            phase("json_parse", profile.json_parse, total_measured, records),
            phase("fen_parse", profile.fen_parse, total_measured, records),
            phase(
                "legal_generation",
                profile.legal_generation,
                total_measured,
                records,
            ),
            phase(
                "pseudo_legal_generation",
                profile.pseudo_legal_generation,
                total_measured,
                records,
            ),
            phase(
                "self_check_filter",
                profile.self_check_filter,
                total_measured,
                records,
            ),
            phase(
                "attack_check_local",
                profile.attack_check_local,
                total_measured,
                records,
            ),
            phase(
                "attack_check_pawn",
                profile.attack_check_pawn,
                total_measured,
                records,
            ),
            phase(
                "attack_check_knight",
                profile.attack_check_knight,
                total_measured,
                records,
            ),
            phase(
                "attack_check_king",
                profile.attack_check_king,
                total_measured,
                records,
            ),
            phase(
                "attack_check_slider",
                profile.attack_check_slider,
                total_measured,
                records,
            ),
            phase(
                "attack_check_bishop_ray",
                profile.attack_check_bishop_ray,
                total_measured,
                records,
            ),
            phase(
                "attack_check_rook_ray",
                profile.attack_check_rook_ray,
                total_measured,
                records,
            ),
            phase(
                "legal_move_uci",
                profile.legal_move_uci,
                total_measured,
                records,
            ),
            phase(
                "legal_action_encode",
                profile.legal_action_encode,
                total_measured,
                records,
            ),
            phase(
                "legal_action_sort",
                profile.legal_action_sort,
                total_measured,
                records,
            ),
            phase(
                "legal_action_encoding",
                profile.legal_action_encoding,
                total_measured,
                records,
            ),
            phase(
                "selected_move_resolution",
                profile.selected_move_resolution,
                total_measured,
                records,
            ),
            phase(
                "selected_move_apply",
                profile.selected_move_apply,
                total_measured,
                records,
            ),
            phase(
                "selected_action_encoding",
                profile.selected_action_encoding,
                total_measured,
                records,
            ),
            phase(
                "position_encoding",
                profile.position_encoding,
                total_measured,
                records,
            ),
            phase("annotations", profile.annotations, total_measured, records),
            phase(
                "json_serialize",
                profile.json_serialize,
                total_measured,
                records,
            ),
        ],
        json_sections: json_sections(profile, records),
    }
}

fn json_sections(
    profile: &OracleProfileTotals,
    records: u64,
) -> Vec<(&'static str, JsonSectionReport)> {
    let total_json_bytes = profile.json_serialize_top_level_bytes
        + profile.json_serialize_legal_moves_bytes
        + profile.json_serialize_legal_action_encodings_bytes
        + profile.json_serialize_position_encoding_bytes
        + profile.json_serialize_annotations_bytes;
    vec![
        json_section(
            "top_level",
            profile.json_serialize_top_level_bytes,
            total_json_bytes,
            records,
        ),
        json_section(
            "legal_moves",
            profile.json_serialize_legal_moves_bytes,
            total_json_bytes,
            records,
        ),
        json_section(
            "legal_action_encodings",
            profile.json_serialize_legal_action_encodings_bytes,
            total_json_bytes,
            records,
        ),
        json_section(
            "position_encoding",
            profile.json_serialize_position_encoding_bytes,
            total_json_bytes,
            records,
        ),
        json_section(
            "annotations",
            profile.json_serialize_annotations_bytes,
            total_json_bytes,
            records,
        ),
    ]
}

fn phase(
    name: &'static str,
    duration: Duration,
    total_measured: Duration,
    records: u64,
) -> (&'static str, ProfilePhaseReport) {
    let total_seconds = seconds(total_measured);
    let phase_seconds = seconds(duration);
    (
        name,
        ProfilePhaseReport {
            seconds: phase_seconds,
            milliseconds_per_record: if records == 0 {
                0.0
            } else {
                (phase_seconds * 1_000.0) / records as f64
            },
            share_of_measured: if total_seconds == 0.0 {
                0.0
            } else {
                phase_seconds / total_seconds
            },
        },
    )
}

fn json_section(
    name: &'static str,
    bytes: u64,
    total_json_bytes: u64,
    records: u64,
) -> (&'static str, JsonSectionReport) {
    (
        name,
        JsonSectionReport {
            bytes,
            bytes_per_record: if records == 0 {
                0.0
            } else {
                bytes as f64 / records as f64
            },
            share_of_json_bytes: if total_json_bytes == 0 {
                0.0
            } else {
                bytes as f64 / total_json_bytes as f64
            },
        },
    )
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}
