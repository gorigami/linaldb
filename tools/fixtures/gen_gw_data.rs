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

use ndarray::Array1;
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

    // Synthetic (NOT real GWOSC data -- a deliberately simplified analysis
    // tool) chirp-like template for examples/gw_transient_analysis.lnl's
    // MATCHED_FILTER section: a linearly-swept sine burst spanning roughly
    // GW150914's real final-inspiral frequency range (~35-250 Hz over
    // ~0.2s), Hann-enveloped to avoid the spurious high-frequency content
    // an abrupt on/off would add. NOT a physically accurate post-Newtonian
    // binary-merger waveform (that's its own separate, much larger physics
    // computation, out of scope -- see SIGNAL_PROCESSING_PLAN.md design
    // decision 5) -- a standard-shape stand-in used only to demonstrate the
    // matched-filtering *technique* on real strain data.
    //
    // Placed at the very start of a 4096-sample (1s) buffer (zero
    // elsewhere) so the template's own reference point is sample 0 --
    // MATCHED_FILTER's recovered offset is then exactly its peak lag, no
    // extra arithmetic needed.
    let sample_rate = 4096.0;
    let template_len = 4096usize;
    let chirp_samples = 819usize; // ~0.2s
    let f0 = 35.0;
    let f1 = 250.0;
    let chirp_duration = chirp_samples as f64 / sample_rate;

    let mut template = vec![0.0f32; template_len];
    for (i, sample) in template.iter_mut().enumerate().take(chirp_samples) {
        let t = i as f64 / sample_rate;
        let phase =
            2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * chirp_duration));
        let envelope =
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / chirp_samples as f64).cos());
        *sample = (envelope * phase.sin()) as f32;
    }

    let template_path = "examples/data/gw_strain/chirp_template_1s.h5";
    let template_arr = Array1::from_vec(template);
    let file = hdf5::File::create(template_path)?;
    let ds = file
        .new_dataset::<f32>()
        .shape(template_len)
        .create("template")?;
    ds.write(&template_arr)?;
    eprintln!(
        "Wrote {template_path} (synthetic {f0}-{f1} Hz chirp-like template, {chirp_samples} \
         active samples in a {template_len}-sample buffer, NOT real GWOSC data)"
    );

    // Reusable 0..4095 sample-index fixture: a deterministic integer
    // sequence, needed to label per-sample rows (e.g. a MATCHED_FILTER
    // correlation series) with their original position after a query
    // reorders them -- there's no ROW_NUMBER-over-original-insertion-order
    // primitive, and hand-writing a 4096-value literal in the .lnl script
    // would be impractical. Same role as the 32-value seg_idx literal
    // already inline in the showcase script, just at a scale where
    // generating it once here and loading it via USE DATASET FROM is far
    // more practical than writing it out by hand.
    let sample_index: Vec<f32> = (0..template_len as u32).map(|i| i as f32).collect();
    let index_path = "examples/data/gw_strain/sample_index_4096.h5";
    let index_arr = Array1::from_vec(sample_index);
    let file = hdf5::File::create(index_path)?;
    let ds = file
        .new_dataset::<f32>()
        .shape(template_len)
        .create("idx")?;
    ds.write(&index_arr)?;
    eprintln!(
        "Wrote {index_path} (deterministic 0..{} index fixture)",
        template_len - 1
    );

    eprintln!("Done. Real GWOSC data (LIGO/Virgo Collaborations, CC BY 4.0) written under examples/data/.");
    Ok(())
}
