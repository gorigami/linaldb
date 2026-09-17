//! NetCDF connector integration tests -- SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 1.
//!
//! Builds a synthetic `.nc` file directly via the `hdf5` crate (NetCDF4 files are HDF5 under
//! the hood, and this project has no netCDF-C binding -- writing the same on-disk attribute
//! structure real CF-compliant tooling produces is sufficient to exercise the connector's real
//! read path, not just its unit-tested pure decoding math).

use hdf5::types::VarLenUnicode;
use linal::dsl::persistence;
use std::path::Path;
use std::str::FromStr;

fn write_cf_netcdf(path: &Path) {
    let file = hdf5::File::create(path).expect("create .nc file");

    // "temperature": CF-packed (scale_factor/add_offset), with a fill value and descriptive
    // attributes -- the full CF decoding path.
    let raw: Vec<f32> = vec![0.0, 100.0, -999.0, 50.0];
    let ds = file
        .new_dataset::<f32>()
        .shape(raw.len())
        .create("temperature")
        .expect("create temperature dataset");
    ds.write(&raw).expect("write temperature data");

    ds.new_attr::<f64>()
        .create("scale_factor")
        .expect("create scale_factor attr")
        .write_scalar(&0.1)
        .expect("write scale_factor");
    ds.new_attr::<f64>()
        .create("add_offset")
        .expect("create add_offset attr")
        .write_scalar(&273.15)
        .expect("write add_offset");
    ds.new_attr::<f64>()
        .create("_FillValue")
        .expect("create _FillValue attr")
        .write_scalar(&-999.0)
        .expect("write _FillValue");
    ds.new_attr::<VarLenUnicode>()
        .create("units")
        .expect("create units attr")
        .write_scalar(&VarLenUnicode::from_str("kelvin").unwrap())
        .expect("write units");
    ds.new_attr::<VarLenUnicode>()
        .create("standard_name")
        .expect("create standard_name attr")
        .write_scalar(&VarLenUnicode::from_str("air_temperature").unwrap())
        .expect("write standard_name");

    // "time": no CF attributes at all -- plain passthrough, same length as temperature.
    let time: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let ds2 = file
        .new_dataset::<f32>()
        .shape(time.len())
        .create("time")
        .expect("create time dataset");
    ds2.write(&time).expect("write time data");
}

#[test]
fn nc_extension_routes_to_netcdf_connector_not_hdf5() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.nc");
    write_cf_netcdf(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry
        .find_connector(path.to_str().unwrap())
        .expect("a connector should be found for .nc");
    assert_eq!(
        connector.name(),
        "netcdf",
        ".nc must route to the CF-aware NetCdfConnector, not Hdf5Connector's opaque reading"
    );
}

#[test]
fn h5_and_h5ad_still_route_to_hdf5_connector() {
    // Regression test for the plan's explicit requirement: registering NetCdfConnector before
    // Hdf5Connector must not change routing for the extensions NetCdfConnector doesn't claim
    // (protects notebook 07's real-world .h5ad usage).
    let registry = persistence::get_connector_registry();

    assert!(
        Path::new("test_data.h5").exists(),
        "test_data.h5 must exist (cargo run --example gen_test_data)"
    );
    let h5 = registry
        .find_connector("test_data.h5")
        .expect("a connector should be found for .h5");
    assert_eq!(h5.name(), "hdf5");

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("sample.h5ad");
    hdf5::File::create(&h5ad_path).expect("create empty .h5ad file");
    let h5ad = registry
        .find_connector(h5ad_path.to_str().unwrap())
        .expect("a connector should be found for .h5ad");
    assert_eq!(h5ad.name(), "hdf5");
}

#[test]
fn netcdf_connector_applies_scale_offset_and_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.nc");
    write_cf_netcdf(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let (batch, _lineage) = connector
        .read_dataset(path.to_str().unwrap(), None)
        .expect("should read the .nc file");

    assert_eq!(batch.num_rows(), 4);

    let temp_col = batch
        .column(batch.schema().index_of("temperature").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("temperature should be Float32");

    // raw [0.0, 100.0, -999.0(fill), 50.0], scale=0.1, offset=273.15
    assert!((temp_col.value(0) - 273.15).abs() < 1e-3);
    assert!((temp_col.value(1) - 283.15).abs() < 1e-3);
    assert!(
        temp_col.value(2).is_nan(),
        "fill value should decode to NaN"
    );
    assert!((temp_col.value(3) - 278.15).abs() < 1e-3);

    // "time" has no CF attributes -- pure passthrough.
    let time_col = batch
        .column(batch.schema().index_of("time").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("time should be Float32");
    assert_eq!(time_col.values(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn netcdf_connector_surfaces_units_and_standard_name_as_field_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.nc");
    write_cf_netcdf(&path);

    let registry = persistence::get_connector_registry();
    let connector = registry.find_connector(path.to_str().unwrap()).unwrap();
    let (batch, _lineage) = connector
        .read_dataset(path.to_str().unwrap(), None)
        .unwrap();

    let field = batch
        .schema()
        .field_with_name("temperature")
        .unwrap()
        .clone();
    assert_eq!(
        field
            .metadata()
            .get(linal::core::connectors::netcdf_connector::UNITS_METADATA_KEY),
        Some(&"kelvin".to_string())
    );
    assert_eq!(
        field
            .metadata()
            .get(linal::core::connectors::netcdf_connector::STANDARD_NAME_METADATA_KEY),
        Some(&"air_temperature".to_string())
    );

    // "time" has no units/standard_name attributes -- no metadata keys attached.
    let time_field = batch.schema().field_with_name("time").unwrap().clone();
    assert!(time_field
        .metadata()
        .get(linal::core::connectors::netcdf_connector::UNITS_METADATA_KEY)
        .is_none());
}
