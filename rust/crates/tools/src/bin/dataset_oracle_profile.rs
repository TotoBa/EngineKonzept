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
struct ProfileReport {
    records: u64,
    total_measured_seconds: f64,
    phases: Vec<(&'static str, ProfilePhaseReport)>,
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
                "attack_check_slider",
                profile.attack_check_slider,
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
    }
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

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}
