//! Downloads real gravitational-wave data published by the LIGO/Virgo
//! collaborations via the Gravitational Wave Open Science Center (GWOSC,
//! gwosc.org, data licensed CC BY 4.0):
//!
//! - `examples/data/gwtc1_events.csv`: all 11 GWTC-1-confident confirmed
//!   events with real physical parameters (component masses, chirp mass,
//!   network matched-filter SNR, luminosity distance, effective spin,
//!   redshift, final mass, GPS time) — used as the relational event catalog
//!   in `examples/gw_transient_analysis.lnl`.
//! - `examples/data/gw_strain/<EVENT>_<DETECTOR>.hdf5`: real 4096 Hz, 32
//!   second strain time series (unmodified GWOSC HDF5 files, H1 + L1
//!   detectors) for four physically distinct events spanning the range of
//!   GWTC-1: GW150914 (first detection, high-SNR BBH), GW151226 (lower-mass
//!   BBH), GW170608 (lowest-mass BBH in GWTC-1), GW170817 (binary neutron
//!   star merger, not a black-hole binary at all).
//!
//! Run with `cargo run --example gen_gw_data`.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;

const CATALOG_URL: &str = "https://gwosc.org/eventapi/json/GWTC-1-confident/";
const STRAIN_EVENTS: &[&str] = &["GW150914", "GW151226", "GW170608", "GW170817"];
const STRAIN_DETECTORS: &[&str] = &["H1", "L1"];

fn get_json(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(client.get(url).send()?.error_for_status()?.json()?)
}

fn as_f64_or_empty(v: &Value) -> String {
    match v {
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("linaldb-fixture-generator")
        .build()?;

    eprintln!("Fetching real GWTC-1-confident catalog from {CATALOG_URL} ...");
    let catalog = get_json(&client, CATALOG_URL)?;
    let events = catalog["events"]
        .as_object()
        .ok_or("catalog JSON missing 'events' object")?;

    // Sort by commonName for a stable, human-readable CSV row order.
    let mut by_name: BTreeMap<String, &Value> = BTreeMap::new();
    for (_key, ev) in events {
        let name = ev["commonName"]
            .as_str()
            .ok_or("event missing commonName")?
            .to_string();
        by_name.insert(name, ev);
    }
    assert_eq!(
        by_name.len(),
        11,
        "expected all 11 GWTC-1-confident events, got {}",
        by_name.len()
    );

    fs::create_dir_all("examples/data/gw_strain")?;

    // First pass: fetch per-event strain metadata (GPS start of the 32s
    // file) for the events we'll download strain for, so the CSV can carry
    // the real, fetched file_gps_start rather than a hand-copied constant --
    // the .lnl script computes each event's merger offset within its own
    // 32s file as `gps_time - file_gps_start`, entirely from real data.
    let mut file_gps_start: BTreeMap<String, i64> = BTreeMap::new();
    let mut strain_urls: BTreeMap<(String, String), String> = BTreeMap::new();
    for event in STRAIN_EVENTS {
        let ev = by_name
            .get(*event)
            .unwrap_or_else(|| panic!("{event} not found in fetched catalog"));
        let jsonurl = ev["jsonurl"]
            .as_str()
            .unwrap_or_else(|| panic!("{event} missing jsonurl"));

        eprintln!("Fetching event detail for {event} from {jsonurl} ...");
        let detail = get_json(&client, jsonurl)?;
        let detail_events = detail["events"]
            .as_object()
            .unwrap_or_else(|| panic!("{event} detail JSON missing 'events'"));
        let (_key, detail_ev) = detail_events
            .iter()
            .next()
            .unwrap_or_else(|| panic!("{event} detail JSON has no event entries"));
        let strain_list = detail_ev["strain"]
            .as_array()
            .unwrap_or_else(|| panic!("{event} detail JSON missing 'strain' array"));

        for detector in STRAIN_DETECTORS {
            let entry = strain_list
                .iter()
                .find(|s| {
                    s["detector"].as_str() == Some(*detector)
                        && s["sampling_rate"].as_i64() == Some(4096)
                        && s["format"].as_str() == Some("hdf5")
                })
                .unwrap_or_else(|| {
                    panic!("no 4096 Hz hdf5 strain entry for {event} detector {detector}")
                });
            let url = entry["url"]
                .as_str()
                .unwrap_or_else(|| panic!("strain entry for {event}/{detector} missing url"));
            let gps_start = entry["GPSstart"]
                .as_i64()
                .unwrap_or_else(|| panic!("strain entry for {event}/{detector} missing GPSstart"));

            file_gps_start
                .entry(event.to_string())
                .and_modify(|existing| {
                    assert_eq!(
                        *existing, gps_start,
                        "{event}: H1/L1 strain files disagree on GPSstart"
                    )
                })
                .or_insert(gps_start);
            strain_urls.insert((event.to_string(), detector.to_string()), url.to_string());
        }
    }

    let csv_path = "examples/data/gwtc1_events.csv";
    let mut csv = fs::File::create(csv_path)?;
    // `merger_offset_seconds` is precomputed here in f64 (gps_time -
    // file_gps_start), NOT left for the DSL to compute at query time: LINAL's
    // `Float` is f32 only (there is no true 64-bit float anywhere in the
    // engine, despite `DOUBLE`/`FLOAT64` being accepted CAST/column-type
    // keywords -- see examples/gw_transient_analysis.lnl's precision-limits
    // section). A real GPS time (~1.1e9) already exceeds f32's ~7 significant
    // digits, so `gps_time - file_gps_start` computed inside the DSL loses
    // the sub-second merger offset entirely (e.g. GW150914's real 15.4s
    // becomes an ~64-unit rounding artifact). Precomputing it here keeps the
    // rest of the showcase's "does the loudest segment match the real
    // merger time" check scientifically meaningful.
    writeln!(
        csv,
        "event_name,gps_time,mass_1_source,mass_2_source,chirp_mass_source,network_matched_filter_snr,luminosity_distance,chi_eff,redshift,final_mass_source,file_gps_start,merger_offset_seconds"
    )?;
    for (name, ev) in &by_name {
        let gps_start_field = file_gps_start
            .get(name)
            .map(|g| g.to_string())
            .unwrap_or_default();
        let merger_offset_field = match (file_gps_start.get(name), ev["GPS"].as_f64()) {
            (Some(&gps_start), Some(gps_event)) => {
                format!("{:.3}", gps_event - gps_start as f64)
            }
            _ => String::new(),
        };
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            name,
            as_f64_or_empty(&ev["GPS"]),
            as_f64_or_empty(&ev["mass_1_source"]),
            as_f64_or_empty(&ev["mass_2_source"]),
            as_f64_or_empty(&ev["chirp_mass_source"]),
            as_f64_or_empty(&ev["network_matched_filter_snr"]),
            as_f64_or_empty(&ev["luminosity_distance"]),
            as_f64_or_empty(&ev["chi_eff"]),
            as_f64_or_empty(&ev["redshift"]),
            as_f64_or_empty(&ev["final_mass_source"]),
            gps_start_field,
            merger_offset_field,
        )?;
    }
    eprintln!("Wrote {csv_path} ({} real events)", by_name.len());

    for event in STRAIN_EVENTS {
        for detector in STRAIN_DETECTORS {
            let url = strain_urls
                .get(&(event.to_string(), detector.to_string()))
                .unwrap_or_else(|| panic!("missing fetched url for {event}/{detector}"));
            let out_path = format!("examples/data/gw_strain/{event}_{detector}.hdf5");
            eprintln!("Downloading real strain data: {url} -> {out_path}");
            let bytes = client.get(url).send()?.error_for_status()?.bytes()?;
            fs::write(&out_path, &bytes)?;
            eprintln!("  wrote {} bytes", bytes.len());
        }
    }

    eprintln!("Done. Real GWOSC data (LIGO/Virgo Collaborations, CC BY 4.0) written under examples/data/.");
    Ok(())
}
